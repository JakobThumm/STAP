import dataclasses
import random
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import symbolic
from ctrlutils import eigen
from scipy.spatial.transform import Rotation
from shapely.geometry import LineString, Polygon

from stap.envs.pybullet.sim import math
from stap.envs.pybullet.sim.robot import Robot
from stap.envs.pybullet.table import primitive_actions, utils
from stap.envs.pybullet.table.objects import Box, Hook, Null, Object, Rack, Screwdriver
from stap.envs.pybullet.table.primitives import (
    ACTION_CONSTRAINTS,
    Pick,
    compute_top_down_orientation,
)

dbprint = lambda *args: None  # noqa
# dbprint = print


@dataclasses.dataclass
class Predicate:
    args: List[str]

    @property
    def state_req(self) -> bool:
        "super().value() requires non-null state argument"
        return False

    @property
    def robot_req(self) -> bool:
        "super().value() requires non-null robot argument"
        return False

    @classmethod
    def create(cls, proposition: str) -> "Predicate":
        predicate, args = symbolic.parse_proposition(proposition)
        predicate_classes = {name.lower(): predicate_class for name, predicate_class in globals().items()}
        predicate_class = predicate_classes[predicate]
        return predicate_class(args)

    def sample(self, robot: Robot, objects: Dict[str, Object], state: Sequence["Predicate"]) -> bool:
        """Generates a geometric grounding of a predicate."""
        return True

    def value(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = True,
    ) -> bool:
        """Evaluates to True if the geometrically grounded predicate is satisfied.

        Note (robot, state): Few Predicates require robot and state to evaluate their truth.
        The function signature supports evaluation for predicates that do not, e.g., for
        goal predicate evaluation, while the necessary checks are added to predicates that do.

        Note (sim): When False, the predicate will be evaluated over `objects`' ObjectState.
        When True, the predicate will be evaluated in the current pybullet state, which will
        overwrite ObjectStates and potentially corrupt future sim=False queries. It is recommended to
        use sim=True for checking predicates at environment initialization, and sim=False for planning.
        """
        if (robot is None and self.robot_req) and (state is None and self.state_req):
            raise ValueError(f"{str(self.__class__)}.value() requires robot and state, but None were given.")
        elif robot is None and self.robot_req:
            raise ValueError(f"{str(self.__class__)}.value() requires robot, but None was given.")
        elif state is None and self.state_req:
            raise ValueError(f"{str(self.__class__)}.value() requires state, but None was given.")
        return True

    def get_arg_objects(self, objects: Dict[str, Object]) -> List[Object]:
        return [objects[arg] for arg in self.args]

    def __str__(self) -> str:
        return f"{type(self).__name__.lower()}({', '.join(self.args)})"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other) -> bool:
        return str(self) == str(other)


class HandleGrasp(Predicate):
    """Unary predicate enforcing a handle grasp towards the tail end on a hook object."""

    pass


class UpperHandleGrasp(Predicate):
    """Unary predicate enforcing a handle grasp towards the head on a hook object."""

    pass


class Free(Predicate):
    """Unary predicate enforcing that no top-down occlusions exist on the object."""

    DISTANCE_MIN: Dict[Tuple[Type[Object], Type[Object]], float] = {
        (Box, Box): 0.05,
        (Box, Hook): 0.05,
        (Box, Rack): 0.1,
        (Hook, Rack): 0.1,
    }

    @property
    def state_req(self) -> bool:
        return True

    def value(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = True,
    ) -> bool:
        super().value(objects=objects, robot=robot, state=state)
        if not sim:
            raise ValueError("Free.value() can only be evaluated in sim mode.")

        child_obj = self.get_arg_objects(objects)[0]
        if child_obj.isinstance(Null):
            return True

        for obj in objects.values():
            if f"ingripper({obj})" in state or f"inhand({obj})" in state or obj.isinstance(Null) or obj == child_obj:
                continue
            if utils.is_under(child_obj, obj):
                dbprint(f"{self}.value():", False, f"{child_obj} under {obj}")
                return False

            obj_a, obj_b = sorted((child_obj.type(), obj.type()), key=lambda x: x.__name__)
            try:
                min_distance = Free.DISTANCE_MIN[(obj_a, obj_b)]
            except KeyError:
                continue
            if (
                (obj.isinstance(Rack) and f"beyondworkspace({obj})" in state)
                or f"infront({child_obj}, rack)" in state
                or f"infront({obj}, rack)" in state
            ):
                min_distance = 0.04
            if utils.is_within_distance(child_obj, obj, min_distance, obj.physics_id) and not utils.is_above(
                child_obj, obj
            ):
                dbprint(
                    f"{self}.value():",
                    False,
                    f"{child_obj} and {obj} are within min distance",
                )
                return False

        return True


class Tippable(Predicate):
    """Unary predicate admitting non-upright configurations of an object."""

    pass


class TableBounds:
    """Predicate that specifies minimum and maximum x-y bounds on the table."""

    MARGIN_SCALE: Dict[Type[Object], float] = {Hook: 0.25}

    def get_bounds_and_margin(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the minimum and maximum x-y bounds on the table as well as the modified margins."""
        assert parent_obj.name == "table"

        zone = type(self).__name__.lower()
        poslimit = TableBounds.get_poslimit(child_obj, state)
        if poslimit is not None:
            pos_bounds = poslimit.bounds(child_obj)
            zone = random.choice(list(pos_bounds.keys()))
            # Compute poslimit zone-specific angle
            if f"aligned({child_obj})" in state:
                theta = Aligned.sample_angle(obj=child_obj, zone=zone)
                child_obj.set_pose(utils.compute_object_pose(child_obj, theta))
                margin = utils.compute_margins(child_obj)

            return pos_bounds[zone], margin

        elif f"aligned({child_obj})" in state:
            theta = Aligned.sample_angle(obj=child_obj, zone=zone)
            child_obj.set_pose(utils.compute_object_pose(child_obj, theta))
            margin = utils.compute_margins(child_obj)

        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = utils.TABLE_CONSTRAINTS["table_x_min"]
        xy_max[0] = utils.TABLE_CONSTRAINTS["table_x_max"]
        xy_min[1] = utils.TABLE_CONSTRAINTS["table_y_min"]
        xy_max[1] = utils.TABLE_CONSTRAINTS["table_y_max"]
        xy_min += margin
        xy_max -= margin

        return bounds, margin

    @staticmethod
    def get_poslimit(
        obj: Object,
        state: Sequence[Predicate],
    ) -> Optional["PosLimit"]:
        try:
            idx_prop = state.index(f"poslimit({obj})")
        except ValueError:
            return None
        prop = state[idx_prop]
        assert isinstance(prop, PosLimit)
        return prop

    @classmethod
    def get_zone(
        cls,
        obj: Object,
        state: Sequence[Predicate],
    ) -> Optional["TableBounds"]:
        zones = [prop for prop in state if isinstance(prop, TableBounds) and prop.args[0] == obj]
        if not zones and f"on({obj}, table)" in state:
            return cls()
        elif len(zones) == 1:
            return zones[0]
        elif len(zones) != 1:
            raise ValueError(f"{obj} cannot be in multiple zones: {zones}")

        return None

    @staticmethod
    def scale_margin(obj: Object, margins: np.ndarray) -> np.ndarray:
        try:
            bounds = TableBounds.MARGIN_SCALE[obj.type()]
        except KeyError:
            return margins
        return bounds * margins


class Aligned(Predicate):
    """Unary predicate enforcing that the object and world coordinate frames align."""

    ANGLE_EPS: float = 0.002
    ANGLE_STD: float = 0.05
    ANGLE_ABS: float = 0.1
    ZONE_ANGLES: Dict[Tuple[Type[Object], Optional[str]], float] = {
        (Rack, "inworkspace"): 0.5 * np.pi,
        (Rack, "beyondworkspace"): 0.0,
    }

    @staticmethod
    def sample_angle(obj: Object, zone: Optional[str] = None) -> float:
        angle = 0.0
        while abs(angle) < Aligned.ANGLE_EPS:
            angle = np.random.randn() * Aligned.ANGLE_STD

        try:
            angle_mu = Aligned.ZONE_ANGLES[(obj.type(), zone)]
        except KeyError:
            angle_mu = 0.0

        angle = np.clip(
            angle + angle_mu,
            angle_mu - Aligned.ANGLE_ABS,
            angle_mu + Aligned.ANGLE_ABS,
        )
        angle = (angle + np.pi) % (2 * np.pi) - np.pi

        return angle


class PosLimit(Predicate):
    """Unary predicate limiting the placement positions of particular object types."""

    POS_EPS: Dict[Type[Object], float] = {Rack: 0.01}
    POS_SPEC: Dict[Type[Object], Dict[str, np.ndarray]] = {
        Rack: {
            "inworkspace": np.array([0.44, -0.33]),
            "beyondworkspace": np.array([0.82, 0.00]),
        }
    }

    def bounds(self, child_obj: Object) -> Dict[str, np.ndarray]:
        assert child_obj.name == self.args[0]
        if child_obj.type() not in PosLimit.POS_SPEC:
            raise ValueError(f"Positions not specified for {child_obj.type()}")

        eps = PosLimit.POS_EPS[child_obj.type()]
        xys = PosLimit.POS_SPEC[child_obj.type()].copy()
        for key, val in xys.items():
            if isinstance(val, list):
                # select one element at random
                elem = np.random.randint(len(val))
                xys[key] = val[elem]
        bounds = {k: np.array([xy - eps, xy + eps]) for k, xy in xys.items()}
        return bounds


class InWorkspace(Predicate, TableBounds):
    """Unary predicate ensuring than an object is in the robot workspace."""

    def value(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = True,
    ) -> bool:
        super().value(objects=objects, robot=robot, state=state)

        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance((Null, Rack)):  # Rack is in workspace by construction.
            return True

        obj_pos = obj.pose(sim=sim).pos[:2]
        distance = float(np.linalg.norm(obj_pos))
        if not utils.is_inworkspace(obj_pos=obj_pos, distance=distance, sim=sim):
            dbprint(f"{self}.value():", False, "- pos:", obj_pos[:2], "distance:", distance)
            return False

        return True

    def get_bounds_and_margin(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the minimum and maximum x-y bounds inside the workspace."""
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        zone = type(self).__name__.lower()
        if f"aligned({child_obj})" in state:
            theta = Aligned.sample_angle(obj=child_obj, zone=zone)
            child_obj.set_pose(utils.compute_object_pose(child_obj, theta))
            margin = utils.compute_margins(child_obj)

        poslimit = TableBounds.get_poslimit(child_obj, state)
        if poslimit is not None:
            return poslimit.bounds(child_obj)[zone], margin

        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = utils.TABLE_CONSTRAINTS["workspace_x_min"]
        xy_max[0] = utils.TABLE_CONSTRAINTS["workspace_max_radius"]
        xy_min += margin
        xy_max -= margin

        return bounds, margin


class InCollisionZone(Predicate, TableBounds):
    """Unary predicate ensuring the object is in the collision zone."""

    def value(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = True,
    ) -> bool:
        super().value(objects=objects, robot=robot, state=state)

        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return True

        obj_pos = obj.pose(sim=sim).pos[:2]
        distance = float(np.linalg.norm(obj_pos))
        if not (
            utils.TABLE_CONSTRAINTS["workspace_x_min"] <= obj_pos[0] < utils.TABLE_CONSTRAINTS["operational_x_min"]
            and distance < utils.TABLE_CONSTRAINTS["workspace_max_radius"]
        ):
            dbprint(f"{self}.value():", False, "- pos:", obj_pos, "distance:", distance)
            return False

        return True

    def get_bounds_and_margin(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        margin = TableBounds.scale_margin(child_obj, margin)
        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = utils.TABLE_CONSTRAINTS["workspace_x_min"]
        xy_max[0] = utils.TABLE_CONSTRAINTS["operational_x_min"]
        xy_min += margin
        xy_max -= margin

        return bounds, margin


class InOperationalZone(Predicate, TableBounds):
    """Unary predicate ensuring the object is in the operational zone."""

    def value(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = True,
    ) -> bool:
        super().value(objects=objects, robot=robot, state=state)

        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return True

        obj_pos = obj.pose(sim=sim).pos[:2]
        distance = float(np.linalg.norm(obj_pos))
        if not (
            utils.TABLE_CONSTRAINTS["operational_x_min"] <= obj_pos[0] < utils.TABLE_CONSTRAINTS["operational_x_max"]
            and distance < utils.TABLE_CONSTRAINTS["workspace_max_radius"]
        ):
            dbprint(f"{self}.value():", False, "- pos:", obj_pos, "distance:", distance)
            return False

        return True

    def get_bounds_and_margin(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        margin = TableBounds.scale_margin(child_obj, margin)
        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = utils.TABLE_CONSTRAINTS["operational_x_min"]
        xy_max[0] = utils.TABLE_CONSTRAINTS["operational_x_max"]
        xy_min += margin
        xy_max -= margin

        return bounds, margin


class InObstructionZone(Predicate, TableBounds):
    """Unary predicate ensuring the object is in the obstruction zone."""

    def value(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = True,
    ) -> bool:
        super().value(objects=objects, robot=robot, state=state)

        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return True

        obj_pos = obj.pose(sim=sim).pos[:2]
        distance = float(np.linalg.norm(obj_pos))
        if not (
            obj_pos[0] >= utils.TABLE_CONSTRAINTS["obstruction_x_min"]
            and distance < utils.TABLE_CONSTRAINTS["workspace_max_radius"]
        ):
            dbprint(f"{self}.value():", False, "- pos:", obj_pos, "distance:", distance)
            return False

        return True

    def get_bounds_and_margin(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        margin = TableBounds.scale_margin(child_obj, margin)
        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = utils.TABLE_CONSTRAINTS["obstruction_x_min"]
        xy_max[0] = utils.TABLE_CONSTRAINTS["workspace_max_radius"]
        xy_min += margin
        xy_max -= margin

        return bounds, margin


class BeyondWorkspace(Predicate, TableBounds):
    """Unary predicate ensuring than an object is in beyond the robot workspace."""

    def value(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = True,
    ) -> bool:
        super().value(objects=objects, robot=robot, state=state)

        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return True

        distance = float(np.linalg.norm(obj.pose(sim=sim).pos[:2]))
        if not utils.is_beyondworkspace(obj=obj, distance=distance, sim=sim):
            dbprint(f"{self}.value():", False, "- distance:", distance)
            return False

        return True

    def get_bounds_and_margin(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the minimum and maximum x-y bounds outside the workspace."""
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        zone = type(self).__name__.lower()
        if f"aligned({child_obj})" in state:
            theta = Aligned.sample_angle(obj=child_obj, zone=zone)
            child_obj.set_pose(utils.compute_object_pose(child_obj, theta))
            margin = utils.compute_margins(child_obj)

        poslimit = TableBounds.get_poslimit(child_obj, state)
        if poslimit is not None:
            return poslimit.bounds(child_obj)[zone], margin

        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        r = utils.TABLE_CONSTRAINTS["workspace_max_radius"]
        xy_min[0] = r * np.cos(np.arcsin(0.5 * (xy_max[1] - xy_min[1]) / r))
        xy_min += margin
        xy_max -= margin

        return bounds, margin


class InOodZone(Predicate, TableBounds):
    """Unary predicate ensuring than an object is in beyond the robot workspace."""

    def value(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = True,
    ) -> bool:
        super().value(objects=objects, robot=robot, state=state)

        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return True

        return True

    def get_bounds_and_margin(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the minimum and maximum x-y bounds outside the workspace."""
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = bounds[0, 0]
        xy_max[0] = utils.TABLE_CONSTRAINTS["table_x_min"]
        xy_min += margin
        xy_max -= margin

        return bounds, margin


class Graspable(Predicate):
    def value(
        self,
        objects: Dict[str, Object],
        robot: Robot,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = True,
    ) -> bool:
        super().value(objects=objects, robot=robot, state=state)

        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return True
        pick_action = self.sample_action(obj)
        command_pos, command_quat = self.generate_command_pose(obj, pick_action)
        pre_pos = np.append(command_pos[:2], ACTION_CONSTRAINTS["max_lift_height"])
        if not self.check_pose(robot, pre_pos, command_quat, positional_precision=0.01, orientational_precision=0.05):
            return False
        if not self.check_pose(
            robot, command_pos, command_quat, positional_precision=0.001, orientational_precision=0.01
        ):
            return False
        return True

    def check_pose(
        self,
        robot: Robot,
        pos: Optional[np.ndarray] = None,
        quat: Optional[Union[eigen.Quaterniond, np.ndarray]] = None,
        pos_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        ori_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        timeout: Optional[float] = None,
        positional_precision: Optional[float] = 1e-3,
        orientational_precision: Optional[float] = None,
        ignore_last_half_rotation: bool = True,
    ) -> bool:
        return robot.arm.set_pose_goal(
            pos=pos,
            quat=quat,
            pos_gains=pos_gains,
            ori_gains=ori_gains,
            timeout=timeout,
            positional_precision=positional_precision,
            orientational_precision=orientational_precision,
            ignore_last_half_rotation=ignore_last_half_rotation,
        )

    def generate_command_pose(self, obj: Object, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a = primitive_actions.PickAction(Pick.scale_action(action))
        a_z = a.pos[2]
        a_z = max(a_z, 0.5 * obj.size[2] + 0.01)
        a.pos[2] = min(a_z, 0.5 * obj.size[2] - 0.01)
        obj_pose = obj.pose()
        obj_quat = eigen.Quaterniond(obj_pose.quat)

        # Compute position.
        command_pos = obj_pose.pos + obj_quat * a.pos

        # Compute orientation.
        command_quat = compute_top_down_orientation(a.theta.item(), obj_quat)
        return command_pos, command_quat

    def sample_action(self, obj: Object) -> np.ndarray:
        if obj.isinstance(Hook):
            hook: Hook = obj  # type: ignore
            pos_handle, pos_head, _ = Hook.compute_link_positions(
                hook.head_length, hook.handle_length, hook.handle_y, hook.radius
            )
            action_range = Pick.Action.range()
            if random.random() < hook.handle_length / (hook.handle_length + hook.head_length):
                # Handle.
                random_x = np.random.uniform(*action_range[:, 0])
                pos = np.array([random_x, pos_handle[1], 0])
                theta = 0.0
            else:
                # Head.
                random_y = np.random.uniform(*action_range[:, 1])
                pos = np.array([pos_head[0], random_y, 0])
                theta = np.pi / 2
            pos[2] += 0.015
        elif obj.isinstance(Screwdriver):
            screwdriver: Screwdriver = obj  # type: ignore
            action_range = Pick.Action.range()
            random_x = np.random.uniform(low=-screwdriver.head_length, high=screwdriver.handle_length)
            # random_x = np.random.uniform(low=0.02, high=screwdriver.handle_length)
            if random_x > 0.02:
                pos = np.array([random_x, 0, 0.0])
            else:
                pos = np.array([random_x, 0, 0.00])
            theta = 0.0
        elif obj.isinstance(Box):
            pos = np.array([0.0, 0.0, obj.size[2] / 2.0 - 0.01])
            theta = 0.0  # if random.random() <= 0.5 else np.pi / 2
        else:
            pos = np.array([0.0, 0.0, 0.0])
            theta = 0.0
        vector = np.concatenate([pos, [theta]])
        return vector


class Ingripper(Predicate):
    MAX_GRASP_ATTEMPTS = 1
    SAMPLE_OFFSET = np.array([0.0, 0.0, 0.0])
    # Maximum deviation of the object from the gripper's center y.
    MAX_GRASP_Y_OFFSET = 0.01
    # Gap required between control point and object bottom.
    FINGER_COLLISION_MARGIN = 0.02
    FINGER_WIDTH = 0.022
    FINGER_HEIGHT = -0.01
    FINGER_DISTANCE = 0.08
    THETA_STDDEV = 0.05

    def sample(self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]) -> bool:
        """Samples a geometric grounding of the InGripper(a) predicate."""
        obj = self.get_arg_objects(objects)[0]
        if obj.is_static:
            return True

        # Generate grasp pose.
        for i in range(Ingripper.MAX_GRASP_ATTEMPTS):
            # Get EEF pose
            ee_pose = robot.arm.ee_pose(update=True)
            rot = Rotation.from_quat(ee_pose.quat)
            # Generate a grasp pose in EEF frame.
            grasp_pose = self.generate_grasp_pose(
                obj,
                handlegrasp=f"handlegrasp({obj})" in state,
                upperhandlegrasp=f"upperhandlegrasp({obj})" in state,
            )
            # Transform grasp pose to object frame.
            obj_pose = math.Pose.from_eigen(grasp_pose.to_eigen().inverse())
            rot_obj = rot * Rotation.from_quat(obj_pose.quat)
            obj_pose.quat = Rotation.as_quat(rot_obj)
            obj_pos_offset = rot_obj.apply(obj_pose.pos)
            # Final pos is grasp pose in obj frame + eef pos in world frame.
            obj_pose.pos = obj_pos_offset + ee_pose.pos - robot.arm.ee_offset + Ingripper.SAMPLE_OFFSET

            # Use fake grasp.
            obj.freeze()
            obj.disable_collisions()
            obj.set_pose(obj_pose)
            robot.grasp_object(obj, realistic=False, timeout=10)
            obj.enable_collisions()

            # Make sure object isn't touching gripper.
            obj.unfreeze()
            robot.step_simulation()
            if utils.is_touching(obj, robot):
                break
            elif i + 1 == Ingripper.MAX_GRASP_ATTEMPTS:
                dbprint(f"{self}.sample():", False, "- exceeded max grasp attempts")
                return False

        dbprint(f"{self}.sample():", True)
        return True

    def generate_grasp_pose_hook(
        self, hook: Hook, handlegrasp: bool = False, upperhandlegrasp: bool = False
    ) -> math.Pose:
        """Generate a grasp for the hook."""
        pos_handle, pos_head, pos_joint = Hook.compute_link_positions(
            head_length=hook.head_length,
            handle_length=hook.handle_length,
            handle_y=hook.handle_y,
            radius=hook.radius,
        )
        if (
            handlegrasp
            or upperhandlegrasp
            or np.random.random() < hook.handle_length / (hook.handle_length + hook.head_length)
        ):
            # Handle.
            min_xyz, max_xyz = np.array(hook.bbox)

            if upperhandlegrasp:
                min_xyz[0] = 0.0
            min_xyz[1] = pos_handle[1] - self.MAX_GRASP_Y_OFFSET
            min_xyz[2] += self.FINGER_COLLISION_MARGIN

            max_xyz[0] = pos_head[0] - hook.radius - 0.5 * self.FINGER_WIDTH
            if handlegrasp:
                max_xyz[0] = 0.0
            max_xyz[1] = pos_handle[1] + self.MAX_GRASP_Y_OFFSET

            theta = 0.0
        else:
            # Head.
            min_xyz, max_xyz = np.array(hook.bbox)

            min_xyz[0] = pos_head[0] - self.MAX_GRASP_Y_OFFSET
            if hook.handle_y < 0:
                min_xyz[1] = pos_handle[1] + hook.radius + 0.5 * self.FINGER_WIDTH
            min_xyz[2] += self.FINGER_COLLISION_MARGIN

            max_xyz[0] = pos_head[0] + self.MAX_GRASP_Y_OFFSET
            if hook.handle_y > 0:
                max_xyz[1] = pos_handle[1] - hook.radius - 0.5 * self.FINGER_WIDTH

            temp_min = min_xyz[0]
            temp_max = max_xyz[0]
            min_xyz[0] = min_xyz[1]
            max_xyz[0] = max_xyz[1]
            min_xyz[1] = temp_min
            max_xyz[1] = temp_max
            theta = np.pi / 2
        min_xyz[2] -= hook.radius
        min_xyz[2] = max(min_xyz[2], -self.FINGER_HEIGHT)
        max_xyz[2] -= hook.radius
        xyz = np.random.uniform(min_xyz, max_xyz)
        theta += np.random.normal(scale=self.THETA_STDDEV)
        theta = np.clip(theta, *primitive_actions.PickAction.RANGES["theta"])
        aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
        return math.Pose(pos=xyz, quat=eigen.Quaterniond(aa).coeffs)

    def generate_grasp_pose_screwdriver(self, screwdriver: Screwdriver) -> math.Pose:
        """Generate a grasp for the screwdriver."""
        X_ROT_STDDEV = 1.0
        Y_ROT_STDDEV = 0.05
        min_xyz, max_xyz = np.array(screwdriver.bbox)
        theta = np.random.choice([0.0, np.pi, -np.pi])
        y_center = 0.5 * (min_xyz[1] + max_xyz[1])
        min_xyz[0] = -screwdriver.head_length
        max_xyz[0] = screwdriver.handle_length
        min_xyz[1] = 0.0
        max_xyz[1] = 0.0
        min_xyz[2] *= 0.5
        max_xyz[2] *= 0.5
        xyz = np.random.uniform(min_xyz, max_xyz)
        x_rot = np.clip(np.random.normal(scale=X_ROT_STDDEV), -np.pi, np.pi)
        y_rot = np.clip(np.random.normal(scale=Y_ROT_STDDEV), -np.pi, np.pi)
        z_rot = np.clip(theta + np.random.normal(scale=self.THETA_STDDEV), -np.pi, np.pi)
        rot = Rotation.from_euler("ZYX", [z_rot, y_rot, x_rot])
        return math.Pose(pos=xyz, quat=rot.as_quat())

    def generate_grasp_pose_box(self, obj: Object):
        """Generates a grasp pose in the object frame of reference."""
        theta = np.random.choice([0.0, np.pi / 2, -np.pi / 2, np.pi, -np.pi])
        min_xyz, max_xyz = 0.5 * np.array(obj.bbox)
        min_xyz[2] += self.FINGER_COLLISION_MARGIN
        min_xyz[2] = max(min_xyz[2], max_xyz[0] - self.FINGER_HEIGHT)
        max_xyz[2] -= self.FINGER_COLLISION_MARGIN
        xyz = np.random.uniform(min_xyz, max_xyz)
        theta += np.random.normal(scale=self.THETA_STDDEV)
        theta = np.clip(theta, *primitive_actions.PickAction.RANGES["theta"])
        aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
        return math.Pose(pos=xyz, quat=eigen.Quaterniond(aa).coeffs)

    def generate_grasp_pose(self, obj: Object, handlegrasp: bool = False, upperhandlegrasp: bool = False) -> math.Pose:
        """Generates a grasp pose in the object frame of reference."""
        if obj.isinstance(Hook):
            hook: Hook = obj  # type: ignore
            return self.generate_grasp_pose_hook(hook, handlegrasp, upperhandlegrasp)
        elif obj.isinstance(Screwdriver):
            screwdriver: Screwdriver = obj  # type: ignore
            return self.generate_grasp_pose_screwdriver(screwdriver)
        else:
            # Fit object between gripper fingers.
            return self.generate_grasp_pose_box(obj)

    def value(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = True,
    ) -> bool:
        super().value(objects=objects, robot=robot, state=state)

        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return False

        return utils.is_ingripper(obj, sim=sim)

    def value_simple(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = False,
    ) -> bool:
        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return False

        return utils.is_ingripper(obj, sim=sim)


class Inhand(Predicate):
    def sample(self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]) -> bool:
        """Samples a geometric grounding of the InHand(a) predicate."""
        obj = self.get_arg_objects(objects)[0]
        if obj.is_static:
            return True
        # TODO: Human cannot grasp objects as of now.
        return True

    def value(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = True,
    ) -> bool:
        super().value(objects=objects, robot=robot, state=state)
        raise NotImplementedError("InHand.value() is not yet implemented.")
        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return False

        return utils.is_ingripper(obj, sim=sim)

    def value_simple(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = False,
    ) -> bool:
        raise NotImplementedError("InHand.value() is not yet implemented.")
        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return False

        return utils.is_ingripper(obj, sim=sim)


class Accepting(Predicate):
    """A predicate that indicates if the actor is accepting objects."""

    def sample(self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]) -> bool:
        """Samples a geometric grounding of the InHand(a) predicate."""
        return True

    def value(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = True,
    ) -> bool:
        super().value(objects=objects, robot=robot, state=state)
        for predicate in state:  # type: ignore
            if isinstance(predicate, Inhand):
                return False
        return True

    def value_simple(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = False,
    ) -> bool:
        return self.value(objects=objects, robot=robot, state=state, sim=sim)


class Under(Predicate):
    """Unary predicate enforcing that an object be placed underneath another."""

    def value(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = True,
    ) -> bool:
        super().value(objects=objects, robot=robot, state=state)

        child_obj, parent_obj = self.get_arg_objects(objects)
        if child_obj.isinstance(Null):
            return True

        if not utils.is_under(child_obj, parent_obj, sim=sim):
            dbprint(f"{self}.value():", False)
            return False

        return True

    def value_simple(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = False,
    ) -> bool:
        child_obj, parent_obj = self.get_arg_objects(objects)
        if child_obj.isinstance(Null):
            return True

        if not utils.is_under(child_obj, parent_obj, sim=sim):
            dbprint(f"{self}.value_simple():", False)
            return False

        return True


class InFront(Predicate):
    """Binary predicate enforcing that one object is in-front of another with
    respect to the world x-y coordinate axis."""

    def value(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = True,
    ) -> bool:
        super().value(objects=objects, robot=robot, state=state)

        child_obj, parent_obj = self.get_arg_objects(objects)
        if child_obj.isinstance(Null):
            return True

        child_pos = child_obj.pose(sim=sim).pos
        xy_min, xy_max = parent_obj.aabb(sim=sim)[:, :2]
        if (
            child_pos[0] >= xy_min[0]
            or child_pos[1] <= xy_min[1]
            or child_pos[1] >= xy_max[1]
            or utils.is_under(child_obj, parent_obj, sim=sim)
        ):
            dbprint(f"{self}.value():", False, "- pos:", child_pos)
            return False

        return True

    @staticmethod
    def bounds(
        child_obj: Object,
        parent_obj: Object,
        margin: np.ndarray = np.zeros(2),
    ) -> np.ndarray:
        """Returns the minimum and maximum x-y bounds in front of the parent object."""
        assert parent_obj.isinstance(Rack)

        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_max[0] = xy_min[0]
        xy_min[0] = utils.TABLE_CONSTRAINTS["workspace_x_min"]
        xy_min += margin
        xy_max -= margin

        return bounds


class NonBlocking(Predicate):
    """Binary predicate ensuring that one object is not occupying a straightline
    path from the robot base to another object."""

    PULL_MARGIN: Dict[Tuple[Type[Object], Type[Object]], Dict[Optional[str], float]] = {
        (Box, Rack): {"inworkspace": 3.0, "beyondworkspace": 1.5},
        (Box, Box): {"inworkspace": 3.0, "beyondworkspace": 3.0},
        (Rack, Hook): {"inworkspace": 0.25, "beyondworkspace": 0.25},
    }

    def value(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = True,
    ) -> bool:
        super().value(objects=objects, robot=robot, state=state)

        target_obj, intersect_obj = self.get_arg_objects(objects)
        if target_obj.isinstance(Null) or intersect_obj.isinstance(Null):
            return True

        target_line = LineString([[0, 0], target_obj.pose(sim=sim).pos[:2].tolist()])
        if intersect_obj.isinstance(Hook):
            convex_hulls = Object.convex_hulls(intersect_obj, project_2d=True, sim=sim)
        else:
            convex_hulls = intersect_obj.convex_hulls(world_frame=True, project_2d=True, sim=sim)

        if len(convex_hulls) > 1:
            raise NotImplementedError("Compound shapes are not yet supported")
        vertices = convex_hulls[0]

        try:
            pull_margins = NonBlocking.PULL_MARGIN[(target_obj.type(), intersect_obj.type())]
        except KeyError:
            pull_margins = None

        if pull_margins is not None:
            if utils.is_inworkspace(obj=intersect_obj, sim=sim):
                zone = "inworkspace"
            elif utils.is_beyondworkspace(obj=intersect_obj, sim=sim):
                zone = "beyondworkspace"
            else:
                zone = None
            try:
                margin_scale = pull_margins[zone]
            except KeyError:
                margin_scale = 1
            target_margin = margin_scale * target_obj.size[:2].max()
            # Expand the vertices by the margin.
            center = vertices.mean(axis=0)
            vertices += np.sign(vertices - center) * target_margin

        intersect_poly = Polygon(vertices)
        if intersect_poly.intersects(target_line):
            dbprint(f"{self}.value():", False)
            return False

        return True


class On(Predicate):
    MAX_SAMPLE_ATTEMPTS = 10

    @property
    def state_req(self) -> bool:
        return True

    @property
    def robot_req(self) -> bool:
        return True

    def sample(self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]) -> bool:
        """Samples a geometric grounding of the On(a, b) predicate."""
        child_obj, parent_obj = self.get_arg_objects(objects)

        if child_obj.is_static:
            dbprint(f"{self}.sample():", True, "- static child")
            return True
        if parent_obj.isinstance(Null):
            dbprint(f"{self}.sample():", False, "- null parent")
            return False

        # Parent surface height
        parent_z = parent_obj.aabb()[1, 2] + utils.EPSILONS["aabb"]

        # Generate theta in the world coordinate frame
        if f"aligned({child_obj})" in state:
            theta = Aligned.sample_angle(obj=child_obj)
        else:
            theta = np.random.uniform(-np.pi, np.pi)
        child_obj.set_pose(utils.compute_object_pose(child_obj, theta))

        # Determine object margins after rotating
        margin_world_frame = utils.compute_margins(child_obj)

        try:
            rack_obj = next(obj for obj in objects.values() if obj.isinstance(Rack))
        except StopIteration:
            rack_obj = None

        if parent_obj.name == "table" and rack_obj is not None and f"under({child_obj}, {rack_obj})" in state:
            # Restrict placement location to under the rack
            parent_obj = rack_obj

        # Determine stable sampling regions on parent surface
        if parent_obj.name == "table":
            zone = TableBounds.get_zone(obj=child_obj, state=state)
            if zone is not None:
                bounds, margin_world_frame = zone.get_bounds_and_margin(
                    child_obj=child_obj,
                    parent_obj=parent_obj,
                    state=state,
                    margin=margin_world_frame,
                )
                xy_min, xy_max = bounds

            if rack_obj is not None and f"infront({child_obj}, {rack_obj})" in state:
                infront_bounds = InFront.bounds(child_obj=child_obj, parent_obj=rack_obj, margin=margin_world_frame)
                intersection = self.compute_bound_intersection(bounds, infront_bounds)
                if intersection is None:
                    dbprint(
                        f"{self}.sample():",
                        False,
                        f"- no intersection between infront({child_obj}, {rack_obj}) and {zone}",
                    )
                    return False
                xy_min, xy_max = intersection

        elif parent_obj.isinstance((Rack, Box)):
            xy_min, xy_max = self.compute_stable_region(child_obj, parent_obj)

        else:
            raise ValueError("[Predicate.On] parent object must be a table, rack, or box")

        # Obtain predicates to validate sampled pose
        propositions = [
            prop for prop in state if isinstance(prop, (Free, TableBounds)) and prop.args[-1] == child_obj.name
        ]

        samples = 0
        success = False
        quat_np = child_obj.pose().quat
        T_parent_obj_to_world = parent_obj.pose().to_eigen()
        while not success and samples < len(range(On.MAX_SAMPLE_ATTEMPTS)):
            # Generate pose and convert to world frame (assumes parent in upright)
            quat = eigen.Quaterniond(quat_np)
            xyz_parent_frame = np.zeros(3)
            xyz_parent_frame[:2] = np.random.uniform(xy_min, xy_max)
            xyz_world_frame = T_parent_obj_to_world * xyz_parent_frame
            xyz_world_frame[2] = parent_z + 0.5 * child_obj.size[2]
            if child_obj.isinstance(Rack):
                xyz_world_frame[2] += 0.5 * child_obj.size[2]

            if f"tippable({child_obj})" in state and not child_obj.isinstance((Hook, Rack)):
                # Tip the object over
                if np.random.random() < utils.EPSILONS["tipping"]:
                    axis = np.random.uniform(-1, 1, size=2)
                    axis /= np.linalg.norm(axis)
                    quat = quat * eigen.Quaterniond(eigen.AngleAxisd(np.pi / 2, np.array([*axis, 0.0])))
                    xyz_world_frame[2] = parent_z + 0.8 * child_obj.size[:2].max()

            pose = math.Pose(pos=xyz_world_frame, quat=quat.coeffs)
            child_obj.set_pose(pose)

            if any(not prop.value(robot=robot, objects=objects, state=state) for prop in propositions):
                samples += 1
                continue
            success = True

        dbprint(f"{self}.sample():", success)
        return success

    def value(
        self,
        objects: Dict[str, Object],
        robot: Optional[Robot] = None,
        state: Optional[Sequence["Predicate"]] = None,
        sim: bool = True,
    ) -> bool:
        super().value(objects=objects, robot=robot, state=state)

        child_obj, parent_obj = self.get_arg_objects(objects)
        if child_obj.isinstance(Null):
            return True

        if not utils.is_on(child_obj, parent_obj, sim=sim):
            dbprint(f"{self}.value():", False, "- child below parent")
            return False

        if state is not None:
            if f"tippable({child_obj})" not in state and not utils.is_upright(child_obj, sim=sim):
                dbprint(f"{self}.value():", False, "- child not upright")
                return False

        return True

    def value_simple(self, objects: Dict[str, Object], sim: bool = False) -> bool:
        child_obj, parent_obj = self.get_arg_objects(objects)
        if child_obj.isinstance(Null):
            return True

        return utils.is_on(child_obj, parent_obj, sim=sim)

    @staticmethod
    def compute_stable_region(
        child_obj: Object,
        parent_obj: Object,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Heuristically compute stable placement region on parent object."""
        # Compute child aabb in parent object frame
        R_child_to_world = child_obj.pose().to_eigen().matrix[:3, :3]
        R_world_to_parent = parent_obj.pose().to_eigen().inverse().matrix[:3, :3]
        vertices = np.concatenate(child_obj.convex_hulls(), axis=0).T
        vertices = R_world_to_parent @ R_child_to_world @ vertices
        child_aabb = np.array([vertices.min(axis=1), vertices.max(axis=1)])

        # Compute margin in the parent frame
        margin = 0.5 * np.array([child_aabb[1, 0] - child_aabb[0, 0], child_aabb[1, 1] - child_aabb[0, 1]])
        xy_min = margin
        xy_max = parent_obj.size[:2] - margin
        if np.any(xy_max - xy_min <= 0):
            # Increase the likelihood of a stable placement location
            child_parent_ratio = 2 * margin / parent_obj.size[:2]
            x_min_ratio = min(0.25 * child_parent_ratio[0], 0.45)
            x_max_ratio = max(0.55, min(0.75 * child_parent_ratio[0], 0.95))
            y_min_ratio = min(0.25 * child_parent_ratio[1], 0.45)
            y_max_ratio = max(0.55, min(0.75 * child_parent_ratio[1], 0.95))
            xy_min[:2] = parent_obj.size[:2] * np.array([x_min_ratio, y_min_ratio])
            xy_max[:2] = parent_obj.size[:2] * np.array([x_max_ratio, y_max_ratio])

        xy_min -= 0.5 * parent_obj.size[:2]
        xy_max -= 0.5 * parent_obj.size[:2]
        return xy_min, xy_max

    @staticmethod
    def compute_bound_intersection(*bounds: np.ndarray) -> Optional[np.ndarray]:
        """Compute intersection of a sequence of xy_min and xy_max bounds."""
        stacked_bounds = np.array(bounds)
        xy_min = stacked_bounds[:, 0].max(axis=0)
        xy_max = stacked_bounds[:, 1].min(axis=0)

        if not (xy_max - xy_min > 0).all():
            return None

        return np.array([xy_min, xy_max])


class TPose(Predicate):
    """Predicate to put the human in a T-pose."""

    MAX_SAMPLE_ATTEMPTS = 1

    @property
    def state_req(self) -> bool:
        return True

    @property
    def robot_req(self) -> bool:
        return False

    def sample(self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]) -> bool:
        """Samples a geometric grounding of the On(a, b) predicate."""
        [human] = self.get_arg_objects(objects)

        if human.is_static:
            dbprint(f"{self}.sample():", True, "- static child")
            return True

        human.set_pose(math.Pose(pos=np.array([2, 0, 0]), quat=np.array([0, 0, 0, 1])))
        return True


UNARY_PREDICATES = {
    "handlegrasp": HandleGrasp,
    "upperhandlegrasp": UpperHandleGrasp,
    "free": Free,
    "aligned": Aligned,
    "tippable": Tippable,
    "poslimit": PosLimit,
    "inworkspace": InWorkspace,
    "incollisionzone": InCollisionZone,
    "inoperationalzone": InOperationalZone,
    "inobstructionzone": InObstructionZone,
    "beyondworkspace": BeyondWorkspace,
    "inoodzone": InOodZone,
    "ingripper": Ingripper,
}


BINARY_PREDICATES = {
    "under": Under,
    "infront": InFront,
    "nonblocking": NonBlocking,
    "on": On,
    "graspable": Graspable,
}


PREDICATE_HIERARCHY = [
    "handlegrasp",
    "upperhandlegrasp",
    "free",
    "aligned",
    "tippable",
    "poslimit",
    "inworkspace",
    "incollisionzone",
    "inoperationalzone",
    "inobstructionzone",
    "beyondworkspace",
    "inoodzone",
    "under",
    "infront",
    "nonblocking",
    "on",
    "ingripper",
    "graspable",
]


assert len(UNARY_PREDICATES) + len(BINARY_PREDICATES) == len(PREDICATE_HIERARCHY)


SUPPORTED_PREDICATES = {
    "under(a, b)": Under,
    "on(a, b)": On,
    "ingripper(a)": Ingripper,
    "tpose(a)": TPose,
    "graspable(a)": Graspable,
}
