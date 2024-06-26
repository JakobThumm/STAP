import copy
import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p
import spatialdyn as dyn
from ctrlutils import eigen
from scipy.spatial.transform import Rotation

from stap.envs.pybullet.sim import articulated_body, math, redisgl
from stap.utils.macros import SIMULATION_TIME_STEP


@dataclasses.dataclass
class ArmState:
    """Mutable arm state."""

    pos_des: Optional[np.ndarray] = None
    quat_des: Optional[np.ndarray] = None
    pos_gains: Union[Tuple[float, float], np.ndarray] = (64.0, 16.0)
    ori_gains: Union[Tuple[float, float], np.ndarray] = (64.0, 16.0)
    torque_control: bool = False
    dx_avg: float = 0.0
    w_avg: float = 0.0
    iter_timeout: int = 0


class Arm(articulated_body.ArticulatedBody):
    """Arm controlled with operational space control."""

    def __init__(
        self,
        physics_id: int,
        body_id: int,
        arm_urdf: str,
        torque_joints: List[str],
        q_home: List[float],
        ee_offset: Tuple[float, float, float],
        pos_gains: Tuple[float, float],
        ori_gains: Tuple[float, float],
        nullspace_joint_gains: Tuple[float, float],
        nullspace_joint_indices: List[int],
        pos_threshold: Tuple[float, float],
        ori_threshold: Tuple[float, float],
        timeout: float,
        redisgl_config: Optional[Dict[str, Any]] = None,
        end_effector_id: int = 6,
        lower_limit: Optional[Union[List[float], np.ndarray]] = None,
        upper_limit: Optional[Union[List[float], np.ndarray]] = None,
        joint_ranges_ns: Optional[Union[List[float], np.ndarray]] = None,
    ):
        """Constructs the arm from yaml config.

        Args:
            physics_id: Pybullet physics client id.
            body_id: Pybullet body id.
            arm_urdf: Path to arm-only urdf for spatialdyn. This urdf will be
                used for computing opspace commands.
            torque_joints: List of torque-controlled joint names.
            q_home: Home joint configuration.
            ee_offset: Position offset from last link com to end-effector operational point.
            pos_gains: (kp, kv) position gains.
            ori_gains: (kp, kv) orientation gains.
            nullspace_joint_gains: (kp, kv) nullspace joint gains.
            nullspace_joint_indices: Joints to control in the nullspace.
            pos_threshold: (position, velocity) error threshold for position convergence.
            ori_threshold: (orientation, angular velocity) threshold for orientation convergence.
            timeout: Default command timeout.
            redisgl_config: Config for setting up RedisGl visualization.
            end_effector_id: ID of the end effector.
            lower_limit: Lower joint limits. (lower limits for null space)
            upper_limit: Upper joint limits. (upper limits for null space)
            joint_ranges_ns: Joint ranges for null space.
        """
        super().__init__(
            physics_id=physics_id,
            body_id=body_id,
            torque_joints=torque_joints,
            position_joints=[],
            timeout=timeout,
        )

        self.q_home = np.array(q_home, dtype=np.float64)
        self.ee_offset = np.array(ee_offset, dtype=np.float64)

        self.pos_gains = np.array(pos_gains, dtype=np.float64)
        self.ori_gains = np.array(ori_gains, dtype=np.float64)
        self.nullspace_joint_gains = np.array(nullspace_joint_gains, dtype=np.float64)
        self.nullspace_joint_indices = list(nullspace_joint_indices)

        self.pos_threshold = np.array(pos_threshold, dtype=np.float64)
        self.ori_threshold = np.array(ori_threshold, dtype=np.float64)
        self.end_effector_id = end_effector_id
        self._lower_limit = np.array(lower_limit)
        self._upper_limit = np.array(upper_limit)
        self._joint_ranges_ns = np.array(joint_ranges_ns)

        self._ab = dyn.ArticulatedBody(dyn.urdf.load_model(arm_urdf))
        self.ab.q = self.q_home
        T_home_to_world = dyn.cartesian_pose(self.ab, offset=self.ee_offset)
        self.quat_home = eigen.Quaterniond(T_home_to_world.linear)
        self.home_pose = self.ee_pose(update=False)

        self._q_limits = np.array([self.link(link_id).joint_limits for link_id in self.torque_joints]).T

        self._redisgl = (
            None
            if redisgl_config is None
            else redisgl.RedisGl(**redisgl_config, ee_offset=self.ee_offset, arm_urdf=arm_urdf)
        )

        self._arm_state = ArmState()
        self._prior = self.q_home
        self.reset(time=0.0)

    @property
    def ab(self) -> dyn.ArticulatedBody:
        """Spatialdyn articulated body."""
        return self._ab

    def reset(self, time: Optional[float] = None, qpos: Optional[Union[np.ndarray, List[float]]] = None) -> bool:
        """Disables torque control and resets the arm to the home configuration (bypassing simulation).

        Args:
            time (float): Simulation time.
            qpos (np.ndarray): Joint configuration to reset to. If none, uses the home configuration.
        """
        self._arm_state = ArmState()
        resetting_qpos = self.q_home if qpos is None else np.array(qpos)
        self.set_configuration_goal(resetting_qpos, skip_simulation=True, time=time)
        return True

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
            pos_gains: (kp, kv) gains or [3 x 2] array of xyz gains.
            ori_gains: (kp, kv) gains or [3 x 2] array of xyz gains.
            timeout: Uses the timeout specified in the yaml arm config if None.
            precision: not used here.
            ignore_last_half_rotation: not used here.
            use_prior: Not used here.
        """
        if pos is not None:
            self._arm_state.pos_des = pos
        if quat is not None:
            if isinstance(quat, np.ndarray):
                quat = eigen.Quaterniond(quat)
            quat = quat * self.quat_home.inverse()
            self._arm_state.quat_des = quat.coeffs
        self._arm_state.pos_gains = self.pos_gains if pos_gains is None else pos_gains
        self._arm_state.ori_gains = self.ori_gains if ori_gains is None else ori_gains

        self._arm_state.dx_avg = 1.0
        self._arm_state.w_avg = 1.0
        self.set_timeout(timeout)
        self._arm_state.torque_control = True
        return True

    def get_joint_state(self, joints: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the position and velocities of the given joints.

        Args:
            joints: List of joint ids.
        Returns:
            Joint positions and velocities (q, dq).
        """
        q, dq = super().get_joint_state(joints)
        if self._redisgl is not None:
            self._redisgl.update(q, dq, self._arm_state.pos_des, self._arm_state.quat_des)
        return q, dq

    def set_configuration_goal(
        self, q: np.ndarray, skip_simulation: bool = False, time: Optional[float] = None
    ) -> None:
        """Sets the robot to the desired joint configuration.

        Args:
            q: Joint configuration.
            skip_simulation: Whether to forcibly set the joint positions or use
                torque control to achieve them.
        """
        if skip_simulation:
            self._arm_state.torque_control = False
            self.reset_joints(q, self.torque_joints)
            self.apply_positions(q, self.torque_joints)
            self.ab.q, self.ab.dq = self.get_joint_state(self.torque_joints)
            return

        # TODO: Implement torque control.
        raise NotImplementedError

    def ee_pose(self, update: bool = True) -> math.Pose:
        if update:
            self.ab.q, self.ab.dq = self.get_joint_state(self.torque_joints)
        T_ee_to_world = dyn.cartesian_pose(self.ab, offset=self.ee_offset)
        quat_ee_to_world = eigen.Quaterniond(T_ee_to_world.linear)
        quat_ee = quat_ee_to_world * self.quat_home.inverse()
        return math.Pose(T_ee_to_world.translation, quat_ee.coeffs)

    def check_joint_limits(self, q: np.ndarray, ignore_last_half_rotation: bool = False) -> bool:
        """Checks whether the given joint configuration is within the joint limits.

        Args:
            q: Joint configuration to check.
            ignore_last_half_rotation: Ignores rotation around the last joint that are larger than 180 degrees.
                They are clipped back to the range [-pi, pi] by the modulo(alpha, pi) operator.
        """
        n_joints = q.shape[0]
        last_rot = q[n_joints - 1]
        upper_limit = self._upper_limit[n_joints - 1] if self._upper_limit is not None else np.pi
        if ignore_last_half_rotation and abs(last_rot) > upper_limit:
            # Last joint is rotating end-effector, so we can ignore 180 degree rotation.
            q[n_joints - 1] = last_rot - np.sign(last_rot) * np.pi
        if (
            self._upper_limit is not None
            and self._lower_limit is not None
            and (not np.all(q >= self._lower_limit) or not np.all(q <= self._upper_limit))
        ):
            return False
        return True

    def reset_joint_state(self, q: Union[List[float], np.ndarray]):
        """Resets the joint state to the given configuration."""
        if isinstance(q, list):
            q = np.array(q)
        assert q.shape[0] <= self.q_home.shape[0]
        for i in range(q.shape[0]):
            p.resetJointState(self.body_id, i, q[i])

    def accurate_calculate_inverse_kinematics(
        self,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray] = None,
        positional_precision: Optional[float] = 1e-3,
        orientational_precision: Optional[float] = None,
        max_iter: int = 5,
        prior: Optional[np.ndarray] = None,
        ignore_last_half_rotation: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """Computes the inverse kinematics solution for the given end-effector pose.

        Args:
            target_pos: Desired end-effector position.
            target_quat: Desired end-effector orientation.
            positional_precision: Precision of the solution acceptance in the position.
            orientational_precision: Precision of the solution acceptance in the orientation.
            max_iter: Max. number of iterations to compute the solution.
            prior: Prior joint configuration to start the IK solver from.
            ignore_last_half_rotation: Ignores rotation around the last joint that are larger than 180 degrees.

        Returns:
            Joint configuration and whether the solution was accepted.
        """
        current_q, _ = self.get_joint_state(self.torque_joints)
        use_nullspace = not (self._lower_limit is None or self._upper_limit is None or self._joint_ranges_ns is None)
        n_joints = self.q_home.shape[0]
        rest_pose = self.q_home if use_nullspace else None
        close_enough = False
        iter = 0
        dist2 = 1e30
        desired_q_pos = np.zeros((n_joints,))
        if prior is not None:
            self.reset_joint_state(prior)
        while not close_enough and iter < max_iter:
            desired_q_pos = np.array(
                p.calculateInverseKinematics(
                    self.body_id,
                    self.end_effector_id,
                    target_pos,
                    target_quat,
                    lowerLimits=self._lower_limit,
                    upperLimits=self._upper_limit,
                    jointRanges=self._joint_ranges_ns,
                    restPoses=rest_pose,
                )
            )
            desired_q_pos = desired_q_pos[:n_joints]
            if not self.check_joint_limits(desired_q_pos, ignore_last_half_rotation):
                desired_q_pos = np.clip(desired_q_pos, self._lower_limit, self._upper_limit)
            self.reset_joint_state(desired_q_pos)
            ls = p.getLinkState(self.body_id, self.end_effector_id)
            newPos = ls[4]
            newQuat = ls[5]
            close_enough = (
                np.linalg.norm(target_pos - np.array(newPos)) < positional_precision  # type: ignore
                if positional_precision is not None
                else True
            )
            close_enough = close_enough and (
                np.linalg.norm(target_quat - np.array(newQuat)) < orientational_precision  # type: ignore
                if (orientational_precision is not None and target_quat is not None)
                else True
            )
            iter = iter + 1
        self.reset_joint_state(current_q)
        return desired_q_pos, close_enough

    def inverse_kinematics(
        self,
        pos: np.ndarray,
        quat: Optional[Union[eigen.Quaterniond, np.ndarray]] = None,
        ignore_last_half_rotation: bool = True,
        positional_precision: Optional[float] = 1e-3,
        orientational_precision: Optional[float] = None,
        max_iter: int = 7,
        prior: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool]:
        """Computes a suitable joint configuration to achieve the given end-effector pose.

        Args:
            pos: Desired end-effector position.
            quat: Desired end-effector orientation.
            ignore_last_half_rotation: Ignores rotation around the last joint that are larger than 180 degrees.
                They are clipped back to the range [-pi, pi] by the modulo(alpha, pi) operator.
            precision: Precision of the solution acceptance.
            max_iter: Max. number of iterations to compute the solution.
            prior: Prior joint configuration to start the IK solver from.
        Returns:
            Joint configuration and whether the solution was accepted.
        """
        n_joints = self.q_home.shape[0]
        if isinstance(quat, eigen.Quaterniond):
            rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        elif isinstance(quat, np.ndarray):
            rot = Rotation.from_quat(quat)
        else:
            rot = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
        desired_ee_pos = pos + rot.apply(self.ee_offset)
        rot_world_to_ee = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])
        quat = (rot * rot_world_to_ee).as_quat()
        desired_q_pos, close_enough = self.accurate_calculate_inverse_kinematics(
            target_pos=desired_ee_pos,
            target_quat=quat,
            positional_precision=positional_precision,
            orientational_precision=orientational_precision,
            max_iter=max_iter,
            prior=prior,
            ignore_last_half_rotation=ignore_last_half_rotation,
        )
        if not close_enough and ignore_last_half_rotation and quat is not None:
            # Try the other half rotation.
            other_target_quat = (rot * Rotation.from_euler("XYZ", [0, 0, np.pi]) * rot_world_to_ee).as_quat()
            desired_q_pos, close_enough = self.accurate_calculate_inverse_kinematics(
                target_pos=desired_ee_pos,
                target_quat=other_target_quat,
                positional_precision=positional_precision,
                orientational_precision=orientational_precision,
                max_iter=max_iter,
                prior=prior,
                ignore_last_half_rotation=ignore_last_half_rotation,
            )
        if not close_enough:
            return desired_q_pos, False
        return desired_q_pos, self.check_joint_limits(desired_q_pos, ignore_last_half_rotation)

    def set_timeout(self, timeout: Optional[float] = None) -> None:
        """Sets the timeout for the arm.
        If the timeout is None, choose self.timeout.
        """
        if timeout is None:
            timeout = self.timeout
        self._arm_state.iter_timeout = int(timeout / SIMULATION_TIME_STEP)

    def update_torques(self, time: Optional[float] = None) -> articulated_body.ControlStatus:
        """Computes and applies the torques to control the articulated body to the goal set with `Arm.set_pose_goal().

        Returns:
            Controller status.
        """
        if not self._arm_state.torque_control:
            return articulated_body.ControlStatus.UNINITIALIZED

        self.ab.q, self.ab.dq = self.get_joint_state(self.torque_joints)

        if (self._q_limits[0] >= self.ab.q).any() or (self.ab.q >= self._q_limits[1]).any():
            return articulated_body.ControlStatus.ABORTED

        # Compute torques.
        q_nullspace = np.array(self.ab.q)
        if self._arm_state.pos_des is not None:
            # Assume base joint rotates about z-axis.
            q_nullspace[0] = np.arctan2(self._arm_state.pos_des[1], self._arm_state.pos_des[0])
        q_nullspace[self.nullspace_joint_indices] = self.q_home[self.nullspace_joint_indices]
        tau, status = dyn.opspace_control(
            self.ab,
            pos=self._arm_state.pos_des,
            ori=self._arm_state.quat_des,
            joint=q_nullspace,
            pos_gains=self._arm_state.pos_gains,
            ori_gains=self._arm_state.ori_gains,
            joint_gains=self.nullspace_joint_gains,
            task_pos=self.ee_offset,
            pos_threshold=self.pos_threshold,
            ori_threshold=self.ori_threshold,
        )

        # Abort if singular.
        if status >= 16:
            return articulated_body.ControlStatus.ABORTED

        self.apply_torques(tau)

        # Return positioned converged.
        if status == 15:
            return articulated_body.ControlStatus.POS_CONVERGED

        # Return velocity converged.
        dx_w = dyn.jacobian(self.ab, offset=self.ee_offset).dot(self.ab.dq)
        dx = dx_w[:3]
        w = dx_w[3:]

        self._arm_state.dx_avg = 0.5 * np.sqrt(dx.dot(dx)) + 0.5 * self._arm_state.dx_avg
        self._arm_state.w_avg = 0.5 * np.sqrt(w.dot(w)) + 0.5 * self._arm_state.w_avg
        if self._arm_state.dx_avg < 0.001 and self._arm_state.w_avg < 0.02:
            return articulated_body.ControlStatus.VEL_CONVERGED

        # Return timeout.
        self._arm_state.iter_timeout -= 1
        if self._arm_state.iter_timeout <= 0:
            return articulated_body.ControlStatus.TIMEOUT

        return articulated_body.ControlStatus.IN_PROGRESS

    def set_prior_to_home(self) -> None:
        """Sets the prior to the current joint configuration."""
        self._prior = self.q_home

    def set_prior_to_current(self) -> None:
        """Sets the prior to the current joint configuration."""
        self._prior = self.ab.q

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

    def human_measurement(self, time: Optional[float] = None, human_pos: Optional[np.ndarray] = None) -> None:
        pass

    def visualize(self) -> None:
        pass

    @property
    def visualization_initialized(self) -> bool:
        return True
