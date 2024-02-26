import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rospy
from ctrlutils import eigen
from std_msgs.msg import Float32MultiArray  # noqa

from stap.controllers.joint_space_control import joint_space_control  # noqa: F401
from stap.envs.pybullet.sim import articulated_body
from stap.envs.pybullet.sim.safe_arm import SafeArm as SafeArmSim
from stap.utils.macros import SIMULATION_TIME_STEP


class SafeArm(SafeArmSim):
    """Arm controlled through ROS."""

    def __init__(
        self,
        base_pos: Union[List[float], np.ndarray] = np.zeros(3),
        base_orientation: Union[List[float], np.ndarray] = np.array([0, 0, 0, 1]),
        shield_type: str = "SSM",
        robot_name: str = "panda",
        damping_ratio: float = 1.0,
        max_acceleration: float = 10.0,
        joint_pos_threshold: float = 1e-3,
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
            kwargs: Keyword arguments for `Arm`.
        """
        try:
            rospy.init_node("stap_node")
        except rospy.exceptions.ROSException:
            print("Node has already been initialized, do nothing")
        # Define publisher and subscribers
        # Publish Float 32 array to /sara_shield/goal_joint_pos
        self._goal_joint_pub = rospy.Publisher("/sara_shield/goal_joint_pos", Float32MultiArray, queue_size=100)
        # Subscribe to Float 32 array /sara_shield/observed_joint_pos
        self._joint_pos_sub = rospy.Subscriber(
            "/sara_shield/observed_joint_pos", Float32MultiArray, self.joint_pos_callback
        )

        self._shield_initialized = False
        super().__init__(**kwargs)
        self._redisgl = None
        self._base_pos = base_pos
        self._base_orientation = base_orientation
        self._shield_type = shield_type
        self._joint_pos_threshold = joint_pos_threshold
        self._shield_initialized = True
        self._visualization_initialized = False
        self._human_sphere_viz = []
        self._robot_sphere_viz = []
        self._q = np.zeros((7,))
        self._dq = np.zeros((7,))
        self._q_d = np.zeros((7,))

    def joint_pos_callback(self, data: Float32MultiArray):
        """ROS callback for new joint position."""
        self._q = np.array(data.data)
        # self.reset_joints(self._q, self.torque_joints)
        self.apply_positions(self._q, self.torque_joints)

    def get_joint_state(self, joints: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the position and velocities of the given joints.

        Gets the joint state from the real robot via Redis and applies it to pybullet.

        Args:
            joints: List of joint ids.
        Returns:
            Joint positions and velocities (q, dq).
        """
        assert len(joints) == len(self._q), (
            "Number of joints does not match: " + str(len(joints)) + " != " + str(len(self._q))
        )
        return self._q, self._dq

    def reset_joints(self, q: np.ndarray, joints: List[int]) -> None:
        raise NotImplementedError

    def apply_torques(self, torques: np.ndarray, joints: Optional[List[int]] = None) -> None:
        raise NotImplementedError

    def reset(self, time: Optional[float] = None, qpos: Optional[Union[np.ndarray, List[float]]] = None) -> bool:
        """Disables torque control and resets the arm to the home configuration (bypassing simulation)."""
        super().reset(time=time, qpos=qpos)
        return True

    def reset_shield(self, time: Optional[float] = None) -> bool:
        """Resets the shield to the home configuration (bypassing simulation)."""
        pass

    def human_measurement(self, time: Optional[float] = None, human_pos: Optional[np.ndarray] = None) -> None:
        """Set the human measurement for the shield.

        Args:
            human_pos [n_joints, 3]: Position of the human joints in world frame.
            time: Current simulation time.
        """
        pass

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
        print("Setting joint configuration goal: ", q)
        self._q_d = q
        msg = Float32MultiArray()
        msg.data = q
        self._goal_joint_pub.publish(msg)
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
        print("Setting pose goal: ", pos, quat)
        if pos is None:
            return False
        if quat is None:
            quat = np.array([0, 0, 0, 1])
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
        self.set_configuration_goal(desired_q_pos)
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

        # Wait for sim timestep time.
        rospy.sleep(SIMULATION_TIME_STEP)

        # Only count safe steps towards timeout.
        # TODO implement safety listener
        # if self._shield.get_safety():
        #     self._arm_state.iter_timeout -= 1
        self._arm_state.iter_timeout -= 1
        if np.linalg.norm(self.ab.q - self._q_d) < self._joint_pos_threshold:
            return articulated_body.ControlStatus.POS_CONVERGED

        return articulated_body.ControlStatus.IN_PROGRESS

    def get_state(self) -> Dict[str, Any]:
        state = {
            "articulated_body": super().get_state(),
            "arm": copy.deepcopy(self._arm_state),
        }
        return state
        # raise NotImplementedError

    def set_state(self, state: Dict[str, Any]) -> None:
        super().set_state(state["articulated_body"])
        self._arm_state = copy.deepcopy(state["arm"])
        self.ab.q, self.ab.dq = self.get_joint_state(self.torque_joints)

    def visualize(self) -> None:
        pass

    def init_visualization(self, n_robot_spheres: int, n_human_spheres: int) -> None:
        pass

    @property
    def visualization_initialized(self) -> bool:
        return self._visualization_initialized

    @property
    def base_pos(self) -> np.ndarray:
        return np.array(self._base_pos)

    @property
    def base_orientation(self) -> np.ndarray:
        return np.array(self._base_orientation)
