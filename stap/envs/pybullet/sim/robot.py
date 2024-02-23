import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pybullet as p
import spatialdyn as dyn
from ctrlutils import eigen

from stap.envs import pybullet
from stap.envs.pybullet import real
from stap.envs.pybullet.sim import arm, articulated_body, body, gripper, math
from stap.envs.pybullet.sim.safe_arm import SafeArm
from stap.utils import configs
from stap.utils.macros import SIMULATION_TIME_STEP


class ControlException(Exception):
    """An exception raised due to a control fault (e.g. reaching singularity)."""

    pass


class Robot(body.Body):
    """User-facing robot interface."""

    def __init__(
        self,
        physics_id: int,
        step_simulation_fn: Callable[[], None],
        urdf: str,
        arm_class: Union[str, Type[arm.Arm]],
        arm_kwargs: Dict[str, Any],
        gripper_class: Union[str, Type[gripper.Gripper]],
        gripper_kwargs: Dict[str, Any],
    ):
        """Loads the robot from a urdf file.

        Args:
            physics_id: Pybullet physics client id.
            step_simulation_fn: Function to step simulation.
            urdf: Path to urdf.
            arm_class: In the temporal_policies.envs.pybullet namespace.
            arm_kwargs: Arm kwargs from yaml config.
            gripper_class: In the temporal_policies.envs.pybullet namespace.
            gripper_kwargs: Gripper kwargs from yaml config.
        """
        self._sim_time = None
        body_id = p.loadURDF(
            fileName=urdf,
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER,  # | p.URDF_MERGE_FIXED_LINKS
            physicsClientId=physics_id,
        )
        super().__init__(physics_id, body_id)
        if isinstance(arm_class, str):
            arm_class = configs.get_class(arm_class, pybullet)
        if isinstance(gripper_class, str):
            gripper_class = configs.get_class(gripper_class, pybullet)
        self._arm_class = arm_class
        self._arm = arm_class(physics_id=self.physics_id, body_id=self.body_id, **arm_kwargs)
        T_world_to_ee = dyn.cartesian_pose(self.arm.ab).inverse()
        self._gripper = gripper_class(self.physics_id, self.body_id, T_world_to_ee, **gripper_kwargs)

        self._step_simulation_fn = step_simulation_fn

    @property
    def arm(self) -> arm.Arm:
        """Controllable arm."""
        return self._arm

    @property
    def gripper(self) -> gripper.Gripper:
        """Controllable gripper."""
        return self._gripper

    @property
    def home_pose(self) -> math.Pose:
        return self.arm.home_pose

    def step_simulation(self) -> None:
        """Steps the simulation and sets self._sim_time to the current simulation time."""
        self._sim_time = self._step_simulation_fn()

    def reset(self, time: Optional[float] = None, qpos: Optional[Union[np.ndarray, List[float]]] = None) -> bool:
        """Resets the robot by setting the arm to its home configuration and the gripper to the open position.

        This method disables torque control and bypasses simulation.

        Args:
            time (float): Simulation time.
            qpos (np.ndarray): Joint configuration to reset to. If none, uses the home configuration.
        """
        if self._arm_class == SafeArm and time is None:
            self._sim_time = 0.0
        else:
            self._sim_time = time

        self.gripper.reset()
        self.clear_load()
        status = self.arm.reset(time=self._sim_time, qpos=qpos)
        resetting_qpos = np.array(qpos) if qpos is not None else self.arm.q_home
        if isinstance(self.arm, real.arm.Arm):
            status = self.goto_configuration(resetting_qpos)
        return status

    def clear_load(self) -> None:
        """Resets the end-effector load to the gripper inertia."""
        if self.gripper.inertia is not None:
            self.arm.ab.replace_load(self.gripper.inertia)
        else:
            self.arm.ab.clear_load()

    def set_load(self, inertia: dyn.SpatialInertiad) -> None:
        """Sets the end-effector load to the sum of the given inertia and gripper inertia."""
        if self.gripper.inertia is not None:
            inertia = inertia + self.gripper.inertia
        self.arm.ab.replace_load(inertia)

    def get_state(self) -> Dict[str, Any]:
        return {
            "arm": self.arm.get_state(),
            "gripper": self.gripper.get_state(),
            "load": copy.deepcopy(self.arm.ab.inertia_load),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.arm.set_state(state["arm"])
        self.gripper.set_state(state["gripper"])
        idx_link, load_inertia = next(iter(state["load"].items()))
        self.arm.ab.replace_load(load_inertia, idx_link)

    def goto_home(self) -> bool:
        """Uses opspace control to go to the home position."""
        return self.goto_pose(
            self.home_pose.pos,
            self.home_pose.quat,
            pos_gains=(64, 16),
            ori_gains=(64, 16),
        )

    def _is_colliding(self, body_id_a: int, body_id_b: int, link_id_a: Optional[int] = None) -> bool:
        kwargs = {}
        if link_id_a is not None:
            kwargs["linkIndexA"] = link_id_a
        contacts = p.getContactPoints(bodyA=body_id_a, bodyB=body_id_b, physicsClientId=self.physics_id, **kwargs)

        if not contacts:
            return False

        force = contacts[0][9]
        return force > 0.0

    def goto_pose(
        self,
        pos: Optional[np.ndarray] = None,
        quat: Optional[Union[eigen.Quaterniond, np.ndarray]] = None,
        pos_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        ori_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        timeout: Optional[float] = None,
        check_collisions: Sequence[int] = [],
        check_collision_freq: int = 10,
        positional_precision: Optional[float] = 1e-3,
        orientational_precision: Optional[float] = None,
        ignore_last_half_rotation: bool = True,
    ) -> bool:
        """Uses opspace control to go to the desired pose.

        This method blocks until the command finishes or times out. A
        ControlException will be raised if the grasp controller is aborted.

        Args:
            pos: Optional position. Maintains current position if None.
            quat: Optional quaternion. Maintains current orientation if None.
            pos_gains: (kp, kv) gains or [3 x 2] array of xyz gains.
            ori_gains: (kp, kv) gains or [3 x 2] array of xyz gains.
            timeout: Uses the timeout specified in the yaml arm config if None.
            check_collisions: Raise an exception if the gripper or grasped
                object collides with any of the body_ids in this list.
            check_collision_freq: Iteration interval with which to check
                collisions.
            precision: Precision for IK algorithm.
            ignore_last_half_rotation: Ignores rotation around the last joint that are larger than 180 degrees.
                They are clipped back to the range [-pi, pi] by the modulo(alpha, pi) operator.
        Returns:
            True if the grasp controller converges to the desired position or
            zero velocity, false if the command times out.
        """
        if check_collisions:
            body_ids_a = [self.body_id] * len(self.gripper.finger_links)
            link_ids_a: List[Optional[int]] = list(self.gripper.finger_links)
            grasp_body_id = self.gripper._gripper_state.grasp_body_id
            if grasp_body_id is not None:
                body_ids_a.append(grasp_body_id)
                link_ids_a.append(None)

        # Set the pose goal.
        success = self.arm.set_pose_goal(
            pos=pos,
            quat=quat,
            pos_gains=pos_gains,
            ori_gains=ori_gains,
            timeout=timeout,
            positional_precision=positional_precision,
            orientational_precision=orientational_precision,
            ignore_last_half_rotation=ignore_last_half_rotation,
        )
        if not success:
            raise ControlException(f"Could not resolve inverse kinematics for ({pos}, {quat}).")

        # Simulate until the pose goal is reached.
        status = self.arm.update_torques(self._sim_time)
        self.gripper.update_torques()
        iter = 0
        while status == articulated_body.ControlStatus.IN_PROGRESS:
            self.step_simulation()
            status = self.arm.update_torques(self._sim_time)
            self.gripper.update_torques()
            iter += 1

            if isinstance(self.arm, real.arm.Arm):
                continue

            if not check_collisions or iter % check_collision_freq != 0:
                continue

            # Terminate early if there are collisions with the gripper fingers
            # or grasped object.
            for body_id_a, link_id_a in zip(body_ids_a, link_ids_a):
                for body_id_b in check_collisions:
                    if self._is_colliding(body_id_a, body_id_b, link_id_a):
                        raise ControlException(
                            f"Robot.goto_pose({pos}, {quat}): Collision {body_id_a}:{link_id_a}, {body_id_b}"
                        )
        # print("Robot.goto_pose:", pos, quat, status)

        if status == articulated_body.ControlStatus.ABORTED:
            raise ControlException(f"Robot.goto_pose({pos}, {quat}): Singularity")

        return status in (
            articulated_body.ControlStatus.POS_CONVERGED,
            articulated_body.ControlStatus.VEL_CONVERGED,
        )

    def goto_dynamic_pose(
        self,
        pose_fn: Callable[..., math.Pose],
        termination_fn: Callable[..., bool],
        pos_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        ori_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        timeout: Optional[float] = None,
        positional_precision: Optional[float] = 1e-3,
        orientational_precision: Optional[float] = None,
        update_pose_every: Optional[int] = 5,
        check_collisions: Sequence[int] = [],
        check_collision_freq: int = 10,
    ) -> bool:
        """Uses opspace control to go to the desired pose.

        This method blocks until the command finishes or times out. A
        ControlException will be raised if the grasp controller is aborted.

        Args:
            pose_fn: Function that returns a math.Pose.
            termination_fn: Function that checks if the dynamic pose goal has been reached.
            pos_gains: (kp, kv) gains or [3 x 2] array of xyz gains.
            ori_gains: (kp, kv) gains or [3 x 2] array of xyz gains.
            timeout: Uses the timeout specified in the yaml arm config if None.
            precision: Precision for IK algorithm.
            update_pose_every: Iteration interval with which to update the target pose.
            check_collisions: Raise an exception if the gripper or grasped
                object collides with any of the body_ids in this list.
            check_collision_freq: Iteration interval with which to check
                collisions.
        Returns:
            True if the grasp controller converges to the desired position or
            zero velocity, false if the command times out.
        """
        if check_collisions:
            body_ids_a = [self.body_id] * len(self.gripper.finger_links)
            link_ids_a: List[Optional[int]] = list(self.gripper.finger_links)
            grasp_body_id = self.gripper._gripper_state.grasp_body_id
            if grasp_body_id is not None:
                body_ids_a.append(grasp_body_id)
                link_ids_a.append(None)
        if update_pose_every is None:
            update_pose_every = 10000000000
        # Simulate until the pose goal is reached.
        status = articulated_body.ControlStatus.IN_PROGRESS
        iter = 0
        # self.arm.set_prior_to_current()
        while status == articulated_body.ControlStatus.IN_PROGRESS:
            if iter % update_pose_every == 0:
                pose = pose_fn()
                new_timeout = timeout - iter * SIMULATION_TIME_STEP if timeout is not None else None
                success = self.arm.set_pose_goal(
                    pos=pose.pos,
                    quat=pose.quat,
                    pos_gains=pos_gains,
                    ori_gains=ori_gains,
                    timeout=new_timeout,
                    positional_precision=positional_precision,
                    orientational_precision=orientational_precision,
                    use_prior=True,
                )
            status = self.arm.update_torques(self._sim_time)
            if status == articulated_body.ControlStatus.ABORTED:
                raise ControlException(f"Robot.goto_pose({pose.pos}, {pose.quat}): Singularity")
            if status in (
                articulated_body.ControlStatus.POS_CONVERGED,
                articulated_body.ControlStatus.VEL_CONVERGED,
            ):
                # We don't care about this kind of convergence here.
                status = articulated_body.ControlStatus.IN_PROGRESS
            if termination_fn():
                return True
            self.gripper.update_torques()
            self.step_simulation()
            iter += 1

            if isinstance(self.arm, real.arm.Arm):
                continue

            if not check_collisions or iter % check_collision_freq != 0:
                continue

            # Terminate early if there are collisions with the gripper fingers
            # or grasped object.
            for body_id_a, link_id_a in zip(body_ids_a, link_ids_a):
                for body_id_b in check_collisions:
                    if self._is_colliding(body_id_a, body_id_b, link_id_a):
                        raise ControlException(
                            f"Robot.goto_pose({pose.pos}, {pose.quat}): Collision {body_id_a}:{link_id_a}, {body_id_b}"
                        )
        # print("Robot.goto_pose:", pos, quat, status)
        return False

    def goto_configuration(self, q: np.ndarray) -> bool:
        """Sets the robot to the desired joint configuration.

        Args:
            q: Joint configuration.
        Returns:
            True if the controller converges to the desired position or zero
            velocity, false if the command times out.
        """
        print("Robot.goto_configuration: " + q)
        # Set the configuration goal.
        self.arm.set_configuration_goal(q, time=self._sim_time)
        print("Robot.goto_configuration: " + q)
        # Simulate until the pose goal is reached.
        status = self.arm.update_torques(time=self._sim_time)
        print("status = ", status)
        self.gripper.update_torques()
        print("Gripper updated.")
        while status == articulated_body.ControlStatus.IN_PROGRESS:
            print("Stepping...")
            self.step_simulation()
            status = self.arm.update_torques(time=self._sim_time)
            self.gripper.update_torques()
        print("Finished goto configuration. Status = ", status)
        return status in (
            articulated_body.ControlStatus.POS_CONVERGED,
            articulated_body.ControlStatus.VEL_CONVERGED,
        )

    def wait_for_termination(self, termination_fn: Optional[Callable] = None, timeout: float = 10.0) -> bool:
        """Wait in the current configuration for the termination function to succeed.

        Args:
            termination_fn: Function that checks if the termination condition is met.
            timeout: Timeout in seconds. If timeout is None, default of self.arm.timeout will be used!
        Returns:
            True if the termination function succeeds, false if the timeout is reached.
        """
        self.arm.set_timeout(timeout)
        status = articulated_body.ControlStatus.IN_PROGRESS
        while status != articulated_body.ControlStatus.TIMEOUT:
            self.step_simulation()
            status = self.arm.update_torques(self._sim_time)
            self.gripper.update_torques()
            if termination_fn is not None and termination_fn():
                return True
        return False

    def grasp(
        self,
        command: float,
        pos_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Sets the gripper to the desired grasp (0.0 open, 1.0 closed).

        This method blocks until the command finishes or times out. A
        ControlException will be raised if the grasp controller is aborted.

        Any existing grasp constraints will be cleared and no new ones will be
        created. Use `Robot.grasp_object()` to create a grasp constraint.

        Args:
            command: Desired grasp (range from 0.0 open to 1.0 closed).
            pos_gains: kp gains (only used for sim).
            timeout: Uses the timeout specified in the yaml gripper config if None.
        Returns:
            True if the grasp controller converges to the desired position or
            zero velocity, false if the command times out.
        """
        # Clear any existing grasp constraints.
        self.gripper.remove_grasp_constraint()
        self.clear_load()

        # Set the new grasp command.
        self.gripper.set_grasp(command, pos_gains, timeout)

        # Simulate until the grasp command finishes.
        status = self.gripper.update_torques()
        while status == articulated_body.ControlStatus.IN_PROGRESS:
            self.arm.update_torques(time=self._sim_time)
            status = self.gripper.update_torques()
            self.step_simulation()
        # print("Robot.grasp:", command, status)

        if status == articulated_body.ControlStatus.ABORTED:
            raise ControlException(f"Robot.grasp({command})")

        return status in (
            articulated_body.ControlStatus.POS_CONVERGED,
            articulated_body.ControlStatus.VEL_CONVERGED,
        )

    def grasp_object(
        self,
        obj: body.Body,
        pos_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        timeout: Optional[float] = None,
        realistic: bool = True,
    ) -> bool:
        """Attempts to grasp an object and attaches the object to the gripper via a pose constraint.

        This method blocks until the command finishes or times out. A
        ControlException will be raised if the grasp controller is aborted.

        Args:
            command: Desired grasp (range from 0.0 open to 1.0 closed).
            pos_gains: kp gains (only used for sim).
            timeout: Uses the timeout specified in the yaml gripper config if None.
            realistic: If false, creates a pose constraint regardless of whether
                the object is in a secure grasp.
        Returns:
            True if the object is successfully grasped, false otherwise.
        """
        if realistic:
            self.grasp(1, pos_gains, timeout)

            # Wait for grasped object to settle.
            status = self.gripper.update_torques()
            while (
                status
                in (
                    articulated_body.ControlStatus.VEL_CONVERGED,
                    articulated_body.ControlStatus.IN_PROGRESS,
                )
                and self.gripper._gripper_state.iter_timeout >= 0
                and (obj.twist() > 0.001).any()
            ):
                self.arm.update_torques(time=self._sim_time)
                status = self.gripper.update_torques()
                self.step_simulation()

            # Make sure fingers aren't fully closed.
            if status == articulated_body.ControlStatus.POS_CONVERGED:
                return False
        else:
            self.grasp(1, pos_gains, timeout)
        # Lock the object in place with a grasp constraint.
        if not self.gripper.create_grasp_constraint(obj.body_id, realistic):
            return False

        # Add object load.
        T_obj_to_world = obj.pose().to_eigen()
        T_ee_to_world = dyn.cartesian_pose(self.arm.ab)
        T_obj_to_ee = T_ee_to_world.inverse() * T_obj_to_world
        self.set_load(obj.inertia * T_obj_to_ee)

        return True
