import abc
import random
from functools import partial
from typing import Callable, Dict, List, NamedTuple, Optional, Type, Union

import gym
import numpy as np
from ctrlutils import eigen
from scipy.spatial.transform import Rotation

from stap.envs import base as envs
from stap.envs.pybullet.sim import math
from stap.envs.pybullet.sim.robot import ControlException, Robot
from stap.envs.pybullet.table import object_state, primitive_actions, utils
from stap.envs.pybullet.table.objects import Box, Hook, Null, Object, Rack
from stap.utils.macros import SIMULATION_FREQUENCY, SIMULATION_TIME_STEP

dbprint = lambda *args: None  # noqa
# dbprint = print


ACTION_CONSTRAINTS = {"max_lift_height": 0.35, "max_lift_radius": 0.7}


def compute_top_down_orientation(
    theta: float, quat_obj: eigen.Quaterniond = eigen.Quaterniond.identity()
) -> eigen.Quaterniond:
    """Computes the top-down orientation of the end-effector with respect to a target object.

    Args:
        theta: Angle of the gripper about the world z-axis wrt the target object.
        quat_obj: Orientation of the target object.
    """
    command_aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
    command_quat = quat_obj * eigen.Quaterniond(command_aa)
    return command_quat


def did_object_move(
    obj: Object,
    old_pose: math.Pose,
    max_delta_xyz: float = 0.05,
    max_delta_theta: float = 5.0 * np.pi / 180,
) -> bool:
    """Checks if the object has moved significantly from its old pose."""
    new_pose = obj.pose()
    T_old_to_world = old_pose.to_eigen()
    T_new_to_world = new_pose.to_eigen()
    T_new_to_old = T_old_to_world.inverse() * T_new_to_world

    delta_xyz = float(np.linalg.norm(T_new_to_old.translation))
    delta_theta = eigen.AngleAxisd(eigen.Quaterniond(T_new_to_old.linear)).angle
    return delta_xyz >= max_delta_xyz or delta_theta >= max_delta_theta


def initialize_robot_pose(robot: Robot) -> bool:
    x_min, x_max = (
        utils.TABLE_CONSTRAINTS["table_x_min"],
        ACTION_CONSTRAINTS["max_lift_radius"],
    )
    y_min = utils.TABLE_CONSTRAINTS["table_y_min"]
    y_max = utils.TABLE_CONSTRAINTS["table_y_max"]
    xy_min = np.array([x_min, y_min])
    xy_max = np.array([x_max, y_max])

    while True:
        xy = np.random.uniform(xy_min, xy_max)
        if np.linalg.norm(xy) < ACTION_CONSTRAINTS["max_lift_radius"]:
            break
    theta = np.random.uniform(*object_state.ObjectState.RANGES["wz"])

    pos = np.append(xy, ACTION_CONSTRAINTS["max_lift_height"])
    aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
    quat = eigen.Quaterniond(aa)
    desired_qpos, success = robot.arm.inverse_kinematics(pos, quat, precision=0.01)
    if success:
        robot.reset(qpos=desired_qpos)
        return True
    # try:
    #     robot.goto_pose(pos, quat)
    # except ControlException as e:
    #     dbprint("initialize_robot_pose():\n", e)
    #     return False
    return False


class ExecutionResult(NamedTuple):
    """Return tuple from Primitive.execute()."""

    success: bool
    """Whether the action succeeded."""

    truncated: bool
    """Whether the action was truncated because of a control error."""


class Primitive(envs.Primitive, abc.ABC):
    Action: Type[primitive_actions.PrimitiveAction]

    def __init__(self, env: envs.Env, idx_policy: int, arg_objects: List[Object]):
        super().__init__(env=env, idx_policy=idx_policy)
        self._arg_objects = arg_objects

    @property
    def arg_objects(self) -> List[Object]:
        return self._arg_objects

    @abc.abstractmethod
    def execute(self, action: np.ndarray, real_world: bool = False) -> ExecutionResult:
        """Executes the primitive.

        Args:
            action: Normalized action (inside action_space, not action_scale).
        Returns:
            (success, truncated) 2-tuple.
        """

    def sample(self, uniform: bool = False) -> np.ndarray:
        if not uniform and random.random() < 0.9:
            action = self.normalize_action(self.sample_action().vector)
            action = np.random.normal(loc=action, scale=0.05)
            action = action.astype(np.float32).clip(self.action_space.low, self.action_space.high)
            return action
        else:
            return super().sample()

    @abc.abstractmethod
    def sample_action(self) -> primitive_actions.PrimitiveAction:
        pass

    def get_policy_args(self) -> Optional[Dict[str, List[int]]]:
        """Gets auxiliary policy args for the current primitive.
        Computes the ordered object indices for the given policy.

        The first index is the end-effector, the following indices are the
        primitive arguments (in order), and the remaining indices are for the
        rest of the objects. The last two objects are reserved for the human hands.

        The non-arg objects can be shuffled randomly for training. This method
        also returns the start and end indices of the non-arg objects.

        Returns:
            Dict with `observation_indices` and `shuffle_range` keys.
        """
        from stap.envs.pybullet.human_table_env import HumanTableEnv
        from stap.envs.pybullet.table_env import TableEnv

        assert isinstance(self.env, TableEnv)
        # This should always be the case.
        assert TableEnv.EE_OBSERVATION_IDX == 0
        # Add end-effector index first.
        observation_indices = list(range(TableEnv.MAX_NUM_OBJECTS))
        idx_shuffle_start = 1 + len(self.arg_objects)
        if isinstance(self.env, HumanTableEnv):
            idx_shuffle_end = TableEnv.MAX_NUM_OBJECTS - 2
        else:
            idx_shuffle_end = TableEnv.MAX_NUM_OBJECTS

        return {
            "observation_indices": observation_indices,
            "shuffle_range": [idx_shuffle_start, idx_shuffle_end],
        }
        # return self.env.get_policy_args(self)

    def get_policy_args_ids(self) -> List[int]:
        """Return the index of the policy args in the observation vector."""
        return list(range(1, len(self.arg_objects) + 1))

    def get_non_arg_objects(self, objects: Dict[str, Object]) -> List[Object]:
        """Gets the non-primitive argument objects."""
        return [obj for obj in objects.values() if obj not in self.arg_objects and not obj.isinstance(Null)]

    def create_object_movement_check(
        self,
        non_arg_objects: bool = True,
        arg_objects: bool = False,
        custom_objects: bool = False,
        objects: Optional[Union[List[Object], Dict[str, Object]]] = None,
    ) -> Callable[[], bool]:
        """Returns a function that checks if any non-primitive argument has been significantly perturbed."""
        if sum([non_arg_objects, arg_objects, custom_objects]) != 1:
            raise ValueError("Must specify only one of non_arg_objects, arg_objects, or custom_objects.")

        # Get specified objects.
        if non_arg_objects:
            if objects is None or not isinstance(objects, dict):
                raise ValueError("Require dictionary of objects for non-arg object movement check.")
            objects = self.get_non_arg_objects(objects)
        elif custom_objects:
            if objects is None or not isinstance(objects, list):
                raise ValueError("Require list of objects for custom object movement check.")
        else:
            objects = self.arg_objects

        # Get object poses.
        old_poses = [obj.pose() for obj in objects]

        def did_non_args_move() -> bool:
            """Checks if any object has moved significantly from its old pose."""
            for obj, old_pose in zip(objects, old_poses):
                if obj.is_active:
                    continue
                if did_object_move(obj, old_pose):
                    return True
            return False

        return did_non_args_move

    def __eq__(self, other) -> bool:
        if isinstance(other, Primitive):
            return str(self) == str(other)
        else:
            return False

    def __str__(self) -> str:
        args = "" if self.arg_objects is None else ", ".join(map(str, self.arg_objects))
        return f"{type(self).__name__}({args})"


class Pick(Primitive):
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_scale = gym.spaces.Box(*primitive_actions.PickAction.range())
    Action = primitive_actions.PickAction
    ALLOW_COLLISIONS = False

    def execute(self, action: np.ndarray, real_world: bool = False) -> ExecutionResult:
        from stap.envs.pybullet.table_env import TableEnv

        assert isinstance(self.env, TableEnv)

        # Parse action.
        a = primitive_actions.PickAction(self.scale_action(action))
        dbprint(a)

        # Get object pose.
        obj = self.arg_objects[0]
        obj_pose = obj.pose()
        obj_quat = eigen.Quaterniond(obj_pose.quat)

        # Compute position.
        command_pos = obj_pose.pos + obj_quat * a.pos

        # Compute orientation.
        command_quat = compute_top_down_orientation(a.theta.item(), obj_quat)

        pre_pos = np.append(command_pos[:2], ACTION_CONSTRAINTS["max_lift_height"])

        objects = self.env.objects
        robot = self.env.robot
        allow_collisions = self.ALLOW_COLLISIONS or real_world
        if not allow_collisions:
            did_non_args_move = self.create_object_movement_check(objects=objects)
        try:
            if not real_world and not utils.is_inworkspace(obj=obj):
                raise ControlException(f"Object {obj} is beyond the robot workspace.")

            robot.goto_pose(pre_pos, command_quat, precision=0.01)
            if not allow_collisions and did_non_args_move():
                raise ControlException(f"Robot.goto_pose({pre_pos}, {command_quat}) collided")

            robot.goto_pose(
                command_pos,
                command_quat,
                check_collisions=[obj.body_id for obj in self.get_non_arg_objects(objects)],
                precision=0.002,
            )

            if not robot.grasp_object(obj):
                raise ControlException(f"Robot.grasp_object({obj}) failed")

            robot.goto_pose(pre_pos, command_quat, precision=0.02)
            if not allow_collisions and did_non_args_move():
                raise ControlException(f"Robot.goto_pose({pre_pos}, {command_quat}) collided")
        except ControlException as e:
            dbprint("Pick.execute():\n", e)
            return ExecutionResult(success=False, truncated=True)

        self.env.wait_until_stable()  # handle pick failures
        return ExecutionResult(success=True, truncated=False)

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        obj = self.arg_objects[0]
        if obj.isinstance(Hook):
            hook: Hook = obj  # type: ignore
            pos_handle, pos_head, _ = Hook.compute_link_positions(
                hook.head_length, hook.handle_length, hook.handle_y, hook.radius
            )
            action_range = self.Action.range()
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
        elif obj.isinstance(Box):
            pos = np.array([0.0, 0.0, 0.0])
            theta = 0.0  # if random.random() <= 0.5 else np.pi / 2
        else:
            pos = np.array([0.0, 0.0, 0.0])
            theta = 0.0
        return primitive_actions.PickAction(pos=pos, theta=theta)


class Place(Primitive):
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_scale = gym.spaces.Box(*primitive_actions.PlaceAction.range())
    Action = primitive_actions.PlaceAction
    ALLOW_COLLISIONS = False

    def execute(self, action: np.ndarray, real_world: bool = False, verbose: bool = False) -> ExecutionResult:
        from stap.envs.pybullet.table_env import TableEnv

        assert isinstance(self.env, TableEnv)

        MAX_DROP_DISTANCE = 0.05

        # Parse action.
        a = primitive_actions.PlaceAction(self.scale_action(action))
        dbprint(a)

        obj, target = self.arg_objects

        # Get target pose.
        target_pose = target.pose()
        target_quat = eigen.Quaterniond(target_pose.quat)

        # Scale action to target bbox.
        xy_action_range = primitive_actions.PlaceAction.range()[:, :2]
        xy_normalized = (a.pos[:2] - xy_action_range[0]) / (xy_action_range[1] - xy_action_range[0])
        xy_target_range = np.array(target.bbox[:, :2])
        if target.name == "table":
            xy_target_range[0, 0] = utils.TABLE_CONSTRAINTS["table_x_min"]
            xy_target_range[1, 0] = ACTION_CONSTRAINTS["max_lift_radius"]
        xy_target = (xy_target_range[1] - xy_target_range[0]) * xy_normalized + xy_target_range[0]
        pos = np.append(xy_target, a.pos[2])
        if real_world:
            pos[2] = min(
                self.env.robot.arm.ee_pose().pos[2] - obj.pose().pos[2] + 0.5 * obj.size[2],
                pos[2],
            )

        # Compute position.
        command_pos = target_pose.pos + target_quat * pos

        # Compute orientation.
        if real_world:
            command_quat = compute_top_down_orientation(a.theta.item())
        else:
            command_quat = compute_top_down_orientation(a.theta.item(), target_quat)

        pre_pos = np.append(command_pos[:2], ACTION_CONSTRAINTS["max_lift_height"])

        objects = self.env.objects
        robot = self.env.robot
        allow_collisions = self.ALLOW_COLLISIONS or real_world
        if not allow_collisions:
            did_non_args_move = self.create_object_movement_check(objects=objects)
        try:
            if not real_world and not utils.is_inworkspace(obj_pos=pre_pos[:2]):
                raise ControlException(f"Placement location {pre_pos} is beyond robot workspace.")

            robot.goto_pose(pre_pos, command_quat)
            if not allow_collisions and did_non_args_move():
                if verbose:
                    print("Robot.goto_pose(pre_pos, command_quat) collided")
                raise ControlException(f"Robot.goto_pose({pre_pos}, {command_quat}) collided")

            robot.goto_pose(
                command_pos,
                command_quat,
                check_collisions=[target.body_id] + [obj.body_id for obj in self.get_non_arg_objects(objects)],
            )

            # Make sure object won't drop from too high.
            if not real_world and not utils.is_within_distance(obj, target, MAX_DROP_DISTANCE, robot.physics_id):
                if verbose:
                    print("Object dropped from too high.")
                raise ControlException("Object dropped from too high.")

            robot.grasp(0)
            if not allow_collisions and did_non_args_move():
                if verbose:
                    print("Robot.grasp(0) collided")
                raise ControlException("Robot.grasp(0) collided")

            robot.goto_pose(pre_pos, command_quat)
            if not allow_collisions and did_non_args_move():
                if verbose:
                    print("Robot.goto_pose(pre_pos, command_quat) collided")
                raise ControlException(f"Robot.goto_pose({pre_pos}, {command_quat}) collided")
        except ControlException as e:
            # If robot fails before grasp(0), object may still be grasped.
            dbprint("Place.execute():\n", e)
            if verbose:
                print("Place.execute():\n", e)
            return ExecutionResult(success=False, truncated=True)

        self.env.wait_until_stable()

        if utils.is_below_table(obj):
            # Falling off the table is an exception.
            return ExecutionResult(success=False, truncated=True)

        if not utils.is_upright(obj) or not utils.is_above(obj, target):
            if verbose:
                print("Object not upright or not above target.")
            return ExecutionResult(success=False, truncated=False)

        return ExecutionResult(success=True, truncated=False)

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        action = self.Action.random()
        action_range = action.range()

        # Compute an appropriate place height given the grasped object's height.
        obj = self.arg_objects[0]
        z_gripper = ACTION_CONSTRAINTS["max_lift_height"]
        z_obj = obj.pose().pos[2]
        action.pos[2] = z_gripper - z_obj + 0.5 * obj.size[2]

        action.pos[2] = np.clip(action.pos[2], action_range[0, 2], action_range[1, 2])

        return action

    @classmethod
    def action(cls, action: np.ndarray) -> primitive_actions.PrimitiveAction:
        return primitive_actions.PlaceAction(action)


class Handover(Primitive):
    """Handover primitive.

    The action is three-dimensional, where the first dimension is the pitch angle of the end-effector, the second
    dimension is the distance to the human hand, and third dimension is the height at which the object should be handed
    over. The distance hereby defines the distance in which we switch from a close handover to a far handover.
    The robot will always try to move to a position, such that the end-effector is pointing towards the human hand.
    E.g, pitch=0: e   pitch=pi/2: x <----- e
                  |                 |-----|
                  v                   dist
                  x
    """

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
    action_scale = gym.spaces.Box(*primitive_actions.HandoverAction.range())
    Action = primitive_actions.HandoverAction
    ALLOW_COLLISIONS = False

    def execute(self, action: np.ndarray, real_world: bool = False, verbose: bool = False) -> ExecutionResult:
        from stap.envs.pybullet.human_table_env import HumanTableEnv

        assert isinstance(self.env, HumanTableEnv)

        START_DISTANCE = 0.5
        SUCCESS_DISTANCE = 0.1
        SUCCESS_TIME = 0.5
        WAIT_TIME = 0.1
        TIMEOUT = 10
        ADDITIONAL_OFFSET = np.array([0, 0, 0.2])
        UPDATE_POS_EVERY = 10
        PRECISION = 0.1

        success = False
        self.success_counter = 0
        # Parse action.
        a = primitive_actions.HandoverAction(self.scale_action(action))
        dbprint(a)

        obj, target = self.arg_objects
        command_pose = self.calculate_command_pose(a, target)
        pre_pos = self.env.robot.arm.ee_pose().pos

        objects = self.env.objects
        robot = self.env.robot
        allow_collisions = self.ALLOW_COLLISIONS or real_world
        if not allow_collisions:
            did_non_args_move = self.create_object_movement_check(objects=objects)
        try:
            # if not real_world and not utils.is_inworkspace(obj_pos=pre_pos[:2]):
            #    raise ControlException(f"Placement location {pre_pos} is beyond robot workspace.")
            # If the human hand is too far away, move to home position.
            human_close = self.termination_condition(obj, target, START_DISTANCE, 0.0)
            while not human_close:
                robot.goto_pose(self.env.robot.arm.home_pose.pos, self.env.robot.arm.home_pose.quat)
                human_close = self.termination_condition(obj, target, START_DISTANCE, 0.0)
                self.env.wait_until_stable(
                    min_iters=np.ceil(WAIT_TIME * SIMULATION_FREQUENCY),
                    max_iters=np.ceil(WAIT_TIME * SIMULATION_FREQUENCY),
                )
            # if not allow_collisions and did_non_args_move():
            #    if verbose:
            #        print("Robot.goto_pose(pre_pos, command_quat) collided")
            #    raise ControlException(f"Robot.goto_pose({pre_pos}, {command_quat}) collided")

            pose_fn = partial(self.calculate_command_pose, a, target, ADDITIONAL_OFFSET)
            termination_fn = partial(self.termination_condition, obj, target, SUCCESS_DISTANCE, SUCCESS_TIME)
            robot.arm.set_prior_to_home()
            success = robot.goto_dynamic_pose(
                pose_fn=pose_fn,
                termination_fn=termination_fn,
                update_pose_every=UPDATE_POS_EVERY,
                timeout=TIMEOUT,
                precision=PRECISION,
                check_collisions=[target.body_id] + [obj.body_id for obj in self.get_non_arg_objects(objects)],
            )
            if not success or (not allow_collisions and did_non_args_move()):
                if verbose:
                    print("Robot.goto_dynamic_pose() failed")
                raise ControlException("Robot.goto_dynamic_pose() failed")

            robot.grasp(0)
            # Once the gripper is opened, we assume the handover was a success.
            success = True
            if not allow_collisions and did_non_args_move():
                if verbose:
                    print("Robot.grasp(0) collided")
                raise ControlException("Robot.grasp(0) collided")

            robot.goto_pose(self.env.robot.arm.home_pose.pos, self.env.robot.arm.home_pose.quat)
            if not allow_collisions and did_non_args_move():
                if verbose:
                    print("Robot.goto_pose(pre_pos, command_quat) collided")
                raise ControlException(f"Robot.goto_pose({pre_pos}, {command_quat}) collided")
        except ControlException as e:
            # If robot fails before grasp(0), object may still be grasped.
            dbprint("Place.execute():\n", e)
            if verbose:
                print("Place.execute():\n", e)
            return ExecutionResult(success=False, truncated=True)

        self.env.wait_until_stable()

        return ExecutionResult(success=success, truncated=False)

    def calculate_command_pose(
        self,
        action: primitive_actions.HandoverAction,
        target: Object,
        additional_offset: Optional[np.ndarray] = None,
    ) -> math.Pose:
        if additional_offset is None:
            additional_offset = np.zeros(3)
        target_pos = target.pose().pos
        target_pos[2] = action.height
        base_pos = self.env.robot.arm.base_pos + additional_offset
        human_dist = np.linalg.norm(target_pos - base_pos)
        if human_dist < action.distance:
            return self.command_pose_close(action, target)
        else:
            return self.command_pose_far(action, target, additional_offset)

    def command_pose_close(self, action: primitive_actions.HandoverAction, target: Object) -> math.Pose:
        # Get target pose.
        target_pos = target.pose().pos
        ee_pos = self.env.robot.arm.ee_pose().pos + self.env.robot.arm.ee_offset
        # The yaw angle is defined as the angle between the current end-effector position and the target position on the x-y plane.
        yaw = np.arctan2(target_pos[1] - ee_pos[1], target_pos[0] - ee_pos[0])
        pitch = action.pitch
        distance = 0.05  # action.distance
        vec_to_eef = np.array([0, 0, 1])
        rot = Rotation.from_euler("ZY", [yaw, pitch])  # type: ignore
        command_vec = rot.apply(vec_to_eef)
        command_quat = eigen.Quaterniond(rot.as_quat())
        command_pos = target_pos + command_vec * distance
        return math.Pose(command_pos, command_quat)

    def command_pose_far(
        self,
        action: primitive_actions.HandoverAction,
        target: Object,
        additional_offset: Optional[np.ndarray] = None,
    ) -> math.Pose:
        if additional_offset is None:
            additional_offset = np.zeros(3)
        # Get target pose.
        target_pos = target.pose().pos
        target_pos[2] = action.height
        base_pos = self.env.robot.arm.base_pos + additional_offset
        # The yaw angle is defined as the angle between the current end-effector position and the target position on the x-y plane.
        yaw = np.arctan2(target_pos[1] - base_pos[1], target_pos[0] - base_pos[0])
        pitch = action.pitch
        height = min(target_pos[2] - base_pos[2], action.distance)
        phi = np.arcsin(height / action.distance)
        command_pos = base_pos + np.array(
            [np.cos(phi) * action.distance * np.cos(yaw), np.cos(phi) * action.distance * np.sin(yaw), height]
        )
        rot = Rotation.from_euler("ZY", [yaw, pitch])  # type: ignore
        command_quat = eigen.Quaterniond(rot.as_quat())
        return math.Pose(command_pos, command_quat)

    def termination_condition(
        self, object: Object, target: Object, success_distance: float = 0.1, success_time: float = 0.5
    ) -> bool:
        """Checks if the human hand is within reach of the object for at least `success_time` seconds."""
        target_pos = target.pose().pos
        obj_pos = object.pose().pos
        if np.linalg.norm(target_pos - obj_pos) < success_distance:
            self.success_counter += 1
        else:
            self.success_counter = 0
        if self.success_counter * SIMULATION_TIME_STEP >= success_time:
            return True
        else:
            return False

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        return self.Action.random()

    @classmethod
    def action(cls, action: np.ndarray) -> primitive_actions.PrimitiveAction:
        return primitive_actions.PlaceAction(action)


class StaticHandover(Primitive):
    """Handover primitive that is not adapting to the human.

    The action is three-dimensional, where the first dimension is the pitch angle of the end-effector, the second
    dimension is the distance to the human hand, and third dimension is the height at which the object should be handed
    over. The distance hereby defines the distance in which the EEF should be placed from the base.
    The robot will always try to move to a position, such that the end-effector is pointing towards the human hand.
    E.g, pitch=0: e   pitch=pi/2: x <----- e
                  |                 |-----|
                  v                   dist
                  x
    """

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
    action_scale = gym.spaces.Box(*primitive_actions.HandoverAction.range())
    Action = primitive_actions.HandoverAction
    ALLOW_COLLISIONS = False

    def execute(self, action: np.ndarray, real_world: bool = False, verbose: bool = False) -> ExecutionResult:
        from stap.envs.pybullet.human_table_env import HumanTableEnv

        assert isinstance(self.env, HumanTableEnv)

        SUCCESS_DISTANCE = 0.4
        SUCCESS_TIME = 1.0
        FIRST_MOVEMENT_TIMEOUT = 2.0
        WAIT_TIMEOUT = 15
        ADDITIONAL_OFFSET = np.array([0, 0, 0.2])
        success = False
        self.success_counter = 0
        # Parse action.
        a = primitive_actions.HandoverAction(self.scale_action(action))
        dbprint(a)

        obj, target = self.arg_objects
        objects = self.env.objects
        robot = self.env.robot
        allow_collisions = self.ALLOW_COLLISIONS or real_world
        if not allow_collisions:
            did_non_args_move = self.create_object_movement_check(objects=objects)
        try:
            robot.arm.set_prior_to_home()
            command_pose = self.calculate_command_pose(a, target, ADDITIONAL_OFFSET)
            command_pos = command_pose.pos
            command_quat = command_pose.quat
            success = robot.goto_pose(
                command_pos,
                command_quat,
                check_collisions=[target.body_id] + [obj.body_id for obj in self.get_non_arg_objects(objects)],
                timeout=FIRST_MOVEMENT_TIMEOUT,
                precision=0.1,
            )
            if not success:
                raise ControlException("Moving to handover pose failed")
            termination_fn = partial(self.termination_condition, obj, target, SUCCESS_DISTANCE, SUCCESS_TIME)
            success = robot.wait_for_termination(termination_fn=termination_fn, timeout=WAIT_TIMEOUT)
            if not success:
                raise ControlException("Handover failed: Human hand not within reach")
            obj.freeze()
            robot.grasp(0)
            # Once the gripper is opened, we assume the handover was a success.
            success = True
            if not allow_collisions and did_non_args_move():
                raise ControlException("Robot.grasp(0) collided")

            robot.goto_configuration(robot.arm.q_home)
            if not allow_collisions and did_non_args_move():
                raise ControlException("Robot.goto_pose() collided")
        except ControlException as e:
            # If robot fails before grasp(0), object may still be grasped.
            dbprint("StaticHandover.execute():\n", e)
            if verbose:
                print("StaticHandover.execute():\n", e)
            return ExecutionResult(success=False, truncated=True)

        self.env.wait_until_stable()

        return ExecutionResult(success=success, truncated=False)

    def calculate_command_pose(
        self,
        action: primitive_actions.HandoverAction,
        target: Object,
        additional_offset: Optional[np.ndarray] = None,
    ) -> math.Pose:
        if additional_offset is None:
            additional_offset = np.zeros(3)
        # Get target pose.
        target_pos = target.pose().pos
        target_pos[2] = action.height
        base_pos = self.env.robot.arm.base_pos + additional_offset
        # The yaw angle is defined as the angle between the current end-effector position and the target position on the x-y plane.
        yaw = np.arctan2(target_pos[1] - base_pos[1], target_pos[0] - base_pos[0])
        pitch = action.pitch
        height = min(target_pos[2] - base_pos[2], action.distance)
        phi = np.arcsin(height / action.distance)
        command_pos = base_pos + np.array(
            [np.cos(phi) * action.distance * np.cos(yaw), np.cos(phi) * action.distance * np.sin(yaw), height]
        )
        rot = Rotation.from_euler("ZY", [yaw, pitch])  # type: ignore
        command_quat = eigen.Quaterniond(rot.as_quat())
        return math.Pose(command_pos, command_quat)

    def termination_condition(
        self, object: Object, target: Object, success_distance: float = 0.1, success_time: float = 0.5
    ) -> bool:
        """Checks if the human hand is within reach of the object for at least `success_time` seconds."""
        target_pos = target.pose().pos
        obj_pos = object.pose().pos
        if np.linalg.norm(target_pos - obj_pos) < success_distance:
            self.success_counter += 1
        else:
            self.success_counter = 0
        if self.success_counter * SIMULATION_TIME_STEP >= success_time:
            return True
        else:
            return False

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        return self.Action.random()

    @classmethod
    def action(cls, action: np.ndarray) -> primitive_actions.PrimitiveAction:
        return primitive_actions.PlaceAction(action)


class Pull(Primitive):
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_scale = gym.spaces.Box(*primitive_actions.PullAction.range())
    Action = primitive_actions.PullAction
    ALLOW_COLLISIONS = False

    def execute(self, action: np.ndarray, real_world: bool = False, verbose: bool = False) -> ExecutionResult:
        from stap.envs.pybullet.table_env import TableEnv

        assert isinstance(self.env, TableEnv)

        PULL_HEIGHT = 0.03
        MIN_PULL_DISTANCE = 0.01

        # Parse action.
        a = primitive_actions.PullAction(self.scale_action(action))
        dbprint(a)

        # Get target pose in polar coordinates
        target = self.arg_objects[0]
        hook: Hook = self.arg_objects[1]  # type: ignore
        target_pose = target.pose()
        if target_pose.pos[0] < 0:
            return ExecutionResult(success=False, truncated=True)

        target_distance = np.linalg.norm(target_pose.pos[:2])
        reach_xy = target_pose.pos[:2] / target_distance
        reach_theta = np.arctan2(reach_xy[1], reach_xy[0])
        reach_aa = eigen.AngleAxisd(reach_theta, np.array([0.0, 0.0, 1.0]))
        reach_quat = eigen.Quaterniond(reach_aa)

        target_pos = np.append(target_pose.pos[:2], PULL_HEIGHT)
        T_hook_to_world = hook.pose().to_eigen()
        T_gripper_to_world = self.env.robot.arm.ee_pose().to_eigen()
        T_gripper_to_hook = T_hook_to_world.inverse() * T_gripper_to_world

        # Compute position.
        pos_reach = np.array([a.r_reach, a.y, 0.0])
        hook_pos_reach = target_pos + reach_quat * pos_reach
        pos_pull = np.array([a.r_reach + a.r_pull, a.y, 0.0])
        hook_pos_pull = target_pos + reach_quat * pos_pull

        # Compute orientation.
        hook_quat = compute_top_down_orientation(a.theta.item(), reach_quat)

        T_reach_hook_to_world = math.Pose(hook_pos_reach, hook_quat).to_eigen()
        T_pull_hook_to_world = math.Pose(hook_pos_pull, hook_quat).to_eigen()
        T_reach_to_world = T_reach_hook_to_world * T_gripper_to_hook
        T_pull_to_world = T_pull_hook_to_world * T_gripper_to_hook
        command_pose_reach = math.Pose.from_eigen(T_reach_to_world)
        command_pose_pull = math.Pose.from_eigen(T_pull_to_world)

        pre_pos = np.append(command_pose_reach.pos[:2], ACTION_CONSTRAINTS["max_lift_height"])
        post_pos = np.append(command_pose_pull.pos[:2], ACTION_CONSTRAINTS["max_lift_height"])

        objects = self.env.objects
        robot = self.env.robot
        allow_collisions = self.ALLOW_COLLISIONS or real_world
        if not allow_collisions:
            did_non_args_move = self.create_object_movement_check(objects=objects)
        try:
            robot.goto_pose(pre_pos, command_pose_reach.quat)
            if not allow_collisions and did_non_args_move():
                if verbose:
                    if verbose:
                        print("Robot collided during pre-pose")
                raise ControlException(f"Robot.goto_pose({pre_pos}, {command_pose_reach.quat}) collided")

            robot.goto_pose(
                command_pose_reach.pos,
                command_pose_reach.quat,
                check_collisions=[obj.body_id for obj in self.get_non_arg_objects(objects) if obj.name != "table"],
            )
            if not real_world and not utils.is_upright(target):
                if verbose:
                    print("Target is not upright")
                raise ControlException("Target is not upright", target.pose().quat)

            robot.goto_pose(
                command_pose_pull.pos,
                command_pose_pull.quat,
                pos_gains=np.array([[49, 14], [49, 14], [121, 22]]),
            )
            if not allow_collisions and did_non_args_move():
                if verbose:
                    print("Pull.execute(): collided")
                raise ControlException(f"Robot.goto_pose({command_pose_pull.pos}, {command_pose_pull.quat}) collided")
            if allow_collisions:
                # No objects should move after lifting the hook.
                did_non_args_move = self.create_object_movement_check(objects=objects)

            robot.goto_pose(post_pos, command_pose_pull.quat)
            if did_non_args_move():
                if verbose:
                    print("Pull.execute(): collided 1")
                raise ControlException(f"Robot.goto_pose({post_pos}, {command_pose_pull.quat}) collided")
        except ControlException as e:
            dbprint("Pull.execute():\n", e)
            return ExecutionResult(success=False, truncated=True)

        self.env.wait_until_stable()

        if not real_world and not utils.is_upright(target):
            dbprint("Pull.execute(): not upright")
            if verbose:
                print("Pull.execute(): not upright")
            return ExecutionResult(success=False, truncated=False)

        if not real_world and not utils.is_inworkspace(obj=target):
            dbprint("Pull.execute(): not in workspace")
            if verbose:
                print("Pull.execute(): not in workspace")
            return ExecutionResult(success=False, truncated=False)

        new_target_distance = np.linalg.norm(target.pose().pos[:2])
        if not real_world and new_target_distance >= target_distance - MIN_PULL_DISTANCE:
            if verbose:
                print(f"Pull.execute(): not moved enough {new_target_distance} {target_distance}")
            dbprint("Pull.execute(): not moved enough", new_target_distance, target_distance)
            return ExecutionResult(success=False, truncated=False)

        return ExecutionResult(success=True, truncated=False)

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        action = self.Action.random()

        obj = self.arg_objects[0]
        hook: Hook = self.arg_objects[1]  # type: ignore
        obj_halfsize = 0.5 * np.linalg.norm(obj.size[:2])
        collision_length = 0.5 * hook.size[0] - 2 * hook.radius - obj_halfsize
        action.r_reach = -collision_length
        action.theta = 0.125 * np.pi

        return action


class Push(Primitive):
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_scale = gym.spaces.Box(*primitive_actions.PushAction.range())
    Action = primitive_actions.PushAction
    ALLOW_COLLISIONS = False

    def execute(self, action: np.ndarray, real_world: bool = False) -> ExecutionResult:
        from stap.envs.pybullet.table_env import TableEnv

        assert isinstance(self.env, TableEnv)

        PUSH_HEIGHT = 0.03
        MIN_PUSH_DISTANCE = 0.01

        # Parse action.
        a = primitive_actions.PushAction(self.scale_action(action))
        dbprint(a)

        # Get target pose in polar coordinates
        target = self.arg_objects[0]
        hook: Hook = self.arg_objects[1]  # type: ignore
        target_pose = target.pose()
        if target_pose.pos[0] < 0:
            return ExecutionResult(success=False, truncated=True)

        target_distance = np.linalg.norm(target_pose.pos[:2])
        reach_xy = target_pose.pos[:2] / target_distance
        reach_theta = np.arctan2(reach_xy[1], reach_xy[0])
        reach_aa = eigen.AngleAxisd(reach_theta, np.array([0.0, 0.0, 1.0]))
        reach_quat = eigen.Quaterniond(reach_aa)

        robot = self.env.robot
        target_pos = np.append(target_pose.pos[:2], PUSH_HEIGHT)
        T_hook_to_world = hook.pose().to_eigen()
        T_gripper_to_world = robot.arm.ee_pose().to_eigen()
        T_gripper_to_hook = T_hook_to_world.inverse() * T_gripper_to_world

        # Compute position.
        pos_reach = np.array([a.r_reach, a.y, 0.0])
        hook_pos_reach = target_pos + reach_quat * pos_reach
        pos_push = np.array([a.r_reach + a.r_push, a.y, 0.0])
        hook_pos_push = target_pos + reach_quat * pos_push

        # Compute orientation.
        hook_quat = compute_top_down_orientation(a.theta.item(), reach_quat)

        T_reach_hook_to_world = math.Pose(hook_pos_reach, hook_quat).to_eigen()
        T_push_hook_to_world = math.Pose(hook_pos_push, hook_quat).to_eigen()
        T_reach_to_world = T_reach_hook_to_world * T_gripper_to_hook
        T_push_to_world = T_push_hook_to_world * T_gripper_to_hook
        command_pose_reach = math.Pose.from_eigen(T_reach_to_world)
        command_pose_push = math.Pose.from_eigen(T_push_to_world)
        pre_pos = np.append(command_pose_reach.pos[:2], ACTION_CONSTRAINTS["max_lift_height"])

        objects = self.env.objects
        allow_collisions = self.ALLOW_COLLISIONS or real_world
        if not allow_collisions:
            did_non_args_move = self.create_object_movement_check(
                non_arg_objects=False,
                custom_objects=True,
                objects=[obj for obj in objects.values() if obj.isinstance(Rack)] + self.get_non_arg_objects(objects),
            )
        try:
            robot.goto_pose(pre_pos, command_pose_reach.quat)
            if not allow_collisions and did_non_args_move():
                raise ControlException(f"Robot.goto_pose({pre_pos}, {command_pose_reach.quat}) collided")

            robot.goto_pose(
                command_pose_reach.pos,
                command_pose_reach.quat,
                check_collisions=[obj.body_id for obj in self.get_non_arg_objects(objects)],
            )
            if not utils.is_upright(target):
                raise ControlException("Target is not upright", target.pose().quat)
            if allow_collisions and not real_world:
                # Avoid pushing off the rack.
                did_rack_move = self.create_object_movement_check(
                    non_arg_objects=False,
                    custom_objects=True,
                    objects=[obj for obj in objects.values() if obj.isinstance(Rack)],
                )

            robot.goto_pose(
                command_pose_push.pos,
                command_pose_push.quat,
                pos_gains=np.array([[49, 14], [49, 14], [121, 22]]),
            )
            if (not allow_collisions and did_non_args_move()) or (
                self.ALLOW_COLLISIONS and not real_world and did_rack_move()
            ):
                raise ControlException(f"Robot.goto_pose({command_pose_push.pos}, {command_pose_push.quat}) collided")

            # Target must be pushed a minimum distance.
            new_target_distance = np.linalg.norm(target.pose().pos[:2])
            if new_target_distance <= target_distance + MIN_PUSH_DISTANCE:
                return ExecutionResult(success=False, truncated=True)

            # Target must be pushed underneath rack if it exists.
            if len(self.arg_objects) == 3:
                obj = self.arg_objects[2]
                if obj.isinstance(Rack) and not utils.is_under(target, obj):
                    return ExecutionResult(success=False, truncated=True)

            robot.goto_pose(command_pose_reach.pos, command_pose_reach.quat)

            # Target must be upright.
            if not utils.is_upright(target):
                return ExecutionResult(success=False, truncated=True)

            robot.goto_pose(pre_pos, command_pose_reach.quat)
        except ControlException as e:
            dbprint("Push.execute():\n", e)
            return ExecutionResult(success=False, truncated=True)

        return ExecutionResult(success=True, truncated=False)

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        action = self.Action.random()

        obj = self.arg_objects[0]
        hook: Hook = self.arg_objects[1]  # type: ignore
        obj_halfsize = 0.5 * np.linalg.norm(obj.size[:2])
        collision_length = -0.5 * hook.size[0] - 2 * hook.radius - obj_halfsize
        action.r_reach = collision_length
        action.theta = 0.125 * np.pi

        return action


class Null(Primitive):
    """Null primitive."""

    def __init__(self, env: Optional[envs.Env] = None, arg_objects: Optional[List[str]] = None):
        self._env = env
        if arg_objects is None:
            arg_objects = []
        self._arg_objects = arg_objects

    def scale_action(self, action: primitive_actions.PrimitiveAction) -> primitive_actions.PrimitiveAction:
        return action

    def execute(self, action: primitive_actions.PrimitiveAction, real_world: bool = False) -> ExecutionResult:
        return ExecutionResult(success=True, truncated=False)

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        return np.ones(1)


class Stop(Primitive):
    """Do nothing."""

    def __init__(self, env: envs.Env, arg_objects: Optional[List[str]] = None):
        self._env = env
        if arg_objects is None:
            arg_objects = []
        self._arg_objects = arg_objects

    def scale_action(self, action: primitive_actions.PrimitiveAction) -> primitive_actions.PrimitiveAction:
        return action

    def execute(self, action: primitive_actions.PrimitiveAction, real_world: bool = False) -> ExecutionResult:
        return ExecutionResult(success=True, truncated=False)

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        return np.ones(1)
        return np.ones(1)


PRIMITIVE_MATCHING = {
    "pick": Pick,
    "place": Place,
    "pull": Pull,
    "handover": Handover,
    "static_handover": StaticHandover,
    "push": Push,
    "null": Null,
    "stop": Stop,
}
