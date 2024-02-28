import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import spatialdyn as dyn
from ctrlutils import eigen

import stap.envs.pybullet.sim.body as body
import stap.envs.pybullet.sim.shapes as shapes
from stap.controllers.failsafe_controller import FailsafeController
from stap.controllers.joint_space_control import joint_space_control  # noqa: F401
from stap.envs.pybullet.sim import articulated_body
from stap.envs.pybullet.sim.arm import Arm
from stap.envs.pybullet.sim.math import Pose


class SafeArm(Arm):
    """Arm controlled with operational space control."""

    def __init__(
        self,
        base_pos: Union[List[float], np.ndarray] = np.zeros(3),
        base_orientation: Union[List[float], np.ndarray] = np.array([0, 0, 0, 1]),
        shield_type: str = "SSM",
        robot_name: str = "panda",
        damping_ratio: float = 1.0,
        max_acceleration: float = 10.0,
        joint_pos_threshold: float = 1e-3,
        create_shield_override: bool = False,
        **kwargs,
    ):
        """Constructs the arm from yaml config.

        Args:
            base_pos: Position of the first joint in world frame.
            base_orientation: Orientation of the base in world frame.
            shield_type: Type of shield to use. Options are "OFF", "SSM", and "PFL".
            robot_name: Name of the robot.
            damping_ratio: Damping ratio for the joint space controller.
            max_acceleration: Maximum joint acceleration for the joint space controller.
            joint_pos_threshold: Threshold for joint position convergence.
            create_shield_override: used to not create the shield if we run the real robot setup.
            kwargs: Keyword arguments for `Arm`.
        """
        self._shield_initialized = False
        super().__init__(**kwargs)
        self._redisgl = None
        self._base_pos = base_pos
        self._base_orientation = base_orientation
        self._shield_type = shield_type
        self._damping_ratio = damping_ratio
        self._max_acceleration = max_acceleration
        self._joint_pos_threshold = joint_pos_threshold
        if not create_shield_override:
            self._shield = FailsafeController(
                init_qpos=self.ab.q,
                base_pos=self._base_pos,
                base_orientation=base_orientation,
                shield_type=shield_type,
                robot_name=robot_name,
                kp=self.pos_gains[0],
                damping_ratio=self._damping_ratio,
            )
        self._shield_initialized = True
        self._visualization_initialized = False
        self._human_sphere_viz = []
        self._robot_sphere_viz = []

    def reset(self, time: Optional[float] = None, qpos: Optional[Union[np.ndarray, List[float]]] = None) -> bool:
        """Disables torque control and resets the arm to the home configuration (bypassing simulation)."""
        super().reset(time=time, qpos=qpos)
        # We don't need to reset the safety shield here, because the arm.reset() function
        # calls set_configuration_goal() which resets the shield.
        return True

    def reset_shield(self, time: Optional[float] = None) -> bool:
        """Resets the shield to the home configuration (bypassing simulation)."""
        if not self._shield_initialized:
            return True
        assert time is not None
        q, _ = self.get_joint_state(self.torque_joints)
        self._shield.reset(
            init_qpos=q,
            time=time,
            base_pos=self._base_pos,
            base_orientation=self._base_orientation,
            shield_type=self._shield_type,
        )
        return True

    def human_measurement(self, time: Optional[float] = None, human_pos: Optional[np.ndarray] = None) -> None:
        """Set the human measurement for the shield.

        Args:
            human_pos [n_joints, 3]: Position of the human joints in world frame.
            time: Current simulation time.
        """
        assert time is not None
        if human_pos is not None:
            self._shield.set_human_measurement(human_pos, time)

    def set_configuration_goal(
        self, q: np.ndarray, skip_simulation: bool = False, time: Optional[float] = None
    ) -> None:
        """Sets the robot to the desired joint configuration.

        Args:
            q: Joint configuration.
            skip_simulation: Whether to forcibly set the joint positions or use
                torque control to achieve them.
            time: Simulation time
        """
        # TODO add to arguments later.
        timeout = None
        if skip_simulation:
            assert time is not None
            self._arm_state.torque_control = False
            self.reset_joints(q, self.torque_joints)
            self.apply_positions(q, self.torque_joints)
            self.ab.q, self.ab.dq = self.get_joint_state(self.torque_joints)
            self.reset_shield(time)
        else:
            self._shield.set_goal(q)
            self.set_timeout(timeout)
            self._arm_state.torque_control = True

    def set_pose_goal(
        self,
        pos: Optional[np.ndarray] = None,
        quat: Optional[Union[eigen.Quaterniond, np.ndarray]] = None,
        pos_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        ori_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        timeout: Optional[float] = None,
        positional_precision: Optional[float] = 1e-3,
        orientational_precision: Optional[float] = None,
        ignore_last_half_rotation: bool = True,
        use_prior: bool = False,
    ) -> bool:
        """Sets the pose goal.

        To actually control the robot, call `Arm.update_torques()`.

        Args:
            pos: Optional position. Maintains current position if None.
            quat: Optional quaternion. Maintains current orientation if None.
            pos_gains: Not used here.
            ori_gains: Not used here.
            timeout: Not used here.
            precision: Precision for IK algorithm.
            ignore_last_half_rotation: Ignores rotation around the last joint that are larger than 180 degrees.
                They are clipped back to the range [-pi, pi] by the modulo(alpha, pi) operator.
            use_prior: Use the joint configuration from the last IK cycle to start the IK solver from.
        Returns:
            True if the goal is set successfully.
        """
        if pos is None:
            return False
        if quat is None:
            quat = np.array([0, 0, 0, 1])
        q, _ = self.get_joint_state(self.torque_joints)
        prior = self._prior if use_prior else None
        desired_q_pos, success = self.inverse_kinematics(
            pos=pos,
            quat=quat,
            positional_precision=positional_precision,
            orientational_precision=orientational_precision,
            ignore_last_half_rotation=ignore_last_half_rotation,
            prior=prior,
        )
        if not success:
            return False
        self._prior = desired_q_pos
        # Set the desired joint position as new goal for sara-shield
        self._shield.set_goal(desired_q_pos)
        self.set_timeout(timeout)
        self._arm_state.torque_control = True
        return True

    def update_torques(self, time: Optional[float] = None) -> articulated_body.ControlStatus:
        """Computes and applies the torques to control the articulated body to the goal set with `Arm.set_pose_goal().

        Returns:
            Controller status.
        """
        assert time is not None
        if not self._arm_state.torque_control:
            return articulated_body.ControlStatus.UNINITIALIZED
        # Return timeout.
        if self._arm_state.iter_timeout <= 0:
            return articulated_body.ControlStatus.TIMEOUT
        self.ab.q, self.ab.dq = self.get_joint_state(self.torque_joints)

        if (self._q_limits[0] >= self.ab.q).any() or (self.ab.q >= self._q_limits[1]).any():
            print("Out of joint limits!")
            # return articulated_body.ControlStatus.ABORTED

        # Step sara-shield to get desired pos, vel, acc
        ddq_desired, qpos_desired, qvel_desired = self._shield.step(self.ab.q, self.ab.dq, time)
        # Calculate torques
        # ddq_desired = np.zeros_like(self.ab.q)
        # torques = joint_space_control(ab=self.ab, ddq_desired=ddq_desired, gravity_comp=True)
        gains = (
            np.array([self.pos_gains[0], self._damping_ratio]).T
            if isinstance(self._damping_ratio, list)
            else (self.pos_gains[0], self._damping_ratio)
        )
        torques, converged = dyn.joint_space_control(
            ab=self.ab,
            joint=qpos_desired,
            joint_gains=gains,
            max_joint_acceleration=self._max_acceleration,
            joint_threshold=None,
            gravity_comp=True,
            integration_step=None,
        )

        self.apply_torques(torques)

        # Check if goal joint position is reached
        self.ab.q, self.ab.dq = self.get_joint_state(self.torque_joints)
        # Only count safe steps towards timeout.# Get current orientation.
        if self._shield.get_safety():
            self._arm_state.iter_timeout -= 1
        if (
            np.linalg.norm(self.ab.q - self._shield.goal_qpos) < self._joint_pos_threshold
            and np.linalg.norm(self.ab.q - qpos_desired) < 2e-2
        ):
            return articulated_body.ControlStatus.POS_CONVERGED

        return articulated_body.ControlStatus.IN_PROGRESS

    def get_state(self) -> Dict[str, Any]:
        state = {
            "articulated_body": super().get_state(),
            "arm": copy.deepcopy(self._arm_state),
        }
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        super().set_state(state["articulated_body"])
        self._arm_state = copy.deepcopy(state["arm"])
        self.ab.q, self.ab.dq = self.get_joint_state(self.torque_joints)

    def visualize(self) -> None:
        robot_spheres = self._shield.get_robot_spheres()
        human_spheres = self._shield.get_human_spheres()
        if not self._visualization_initialized:
            self.init_visualization(len(robot_spheres), len(human_spheres))
        for i in range(len(robot_spheres)):
            self._robot_sphere_viz[i].set_pose(Pose(np.array(robot_spheres[i]), np.array([0, 0, 0, 1])))
        for i in range(len(human_spheres)):
            self._human_sphere_viz[i].set_pose(Pose(np.array(human_spheres[i]), np.array([0, 0, 0, 1])))

    def init_visualization(self, n_robot_spheres: int, n_human_spheres: int) -> None:
        assert not self._visualization_initialized
        human_color = [1, 0, 0, 0.5]
        robot_color = [0, 1, 0, 0.5]
        robot_radius = 0.1
        human_radius = 0.1
        for _ in range(n_robot_spheres):
            shape = shapes.Visual(shape=shapes.Sphere(mass=0, color=np.array(robot_color), radius=robot_radius))
            body_id = shapes.create_body(shapes=shape, physics_id=self.physics_id)
            self._robot_sphere_viz.append(body.Body(physics_id=self.physics_id, body_id=body_id))
        for _ in range(n_human_spheres):
            shape = shapes.Visual(shape=shapes.Sphere(mass=0, color=np.array(human_color), radius=human_radius))
            body_id = shapes.create_body(shapes=shape, physics_id=self.physics_id)
            self._human_sphere_viz.append(body.Body(physics_id=self.physics_id, body_id=body_id))
        self._visualization_initialized = True

    @property
    def visualization_initialized(self) -> bool:
        return self._visualization_initialized

    @property
    def base_pos(self) -> np.ndarray:
        return np.array(self._base_pos)

    @property
    def base_orientation(self) -> np.ndarray:
        return np.array(self._base_orientation)
        return np.array(self._base_orientation)
        return np.array(self._base_orientation)
        return np.array(self._base_orientation)
        return np.array(self._base_orientation)
        return np.array(self._base_orientation)
        return np.array(self._base_orientation)
        return np.array(self._base_orientation)
        return np.array(self._base_orientation)
        return np.array(self._base_orientation)
        return np.array(self._base_orientation)
        return np.array(self._base_orientation)
        return np.array(self._base_orientation)
        return np.array(self._base_orientation)
        return np.array(self._base_orientation)
        return np.array(self._base_orientation)
