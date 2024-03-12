from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p
import rospy
from ctrlutils import eigen
from std_msgs.msg import Bool  # noqa

from stap.envs.pybullet.sim import articulated_body
from stap.envs.pybullet.sim import gripper as sim_gripper
from stap.utils.macros import SIMULATION_TIME_STEP


class GripperRos(sim_gripper.Gripper):
    """Gripper controlled with torque control."""

    def __init__(
        self,
        physics_id: int,
        body_id: int,
        T_world_to_ee: eigen.Isometry3d,
        torque_joints: List[str],
        position_joints: List[str],
        finger_links: List[str],
        base_link: str,
        command_multipliers: List[float],
        finger_contact_normals: List[List[float]],
        inertia_kwargs: Dict[str, Any],
        pos_gains: Tuple[float, float],
        pos_threshold: float,
        timeout: float,
        q_home: Optional[List[float]] = None,
    ):
        """Constructs the arm from yaml config.

        Args:
            physics_id: Pybullet physics client id.
            body_id: Pybullet body id.
            torque_joints: List of torque_controlled joint names.
            position_joints: List of position-controlled joint names.
            finger_links: Finger link names.
            base_link: Gripper base link name.
            command_multipliers: Conversion from [0.0, 1.0] grasp command to
                corresponding joint position.
            finger_contact_normals: Direction of the expected contact normal for
                each of the finger links in the finger link frame, pointing
                towards the grasped object.
            inertia_kwargs: Gripper inertia kwargs ({"mass": Float, "com":
                List[Float, 3], "inertia": List[Float, 6]}).
            pos_gains: (kp, kv) position gains.
            pos_threshold: (position, velocity) error threshold for position convergence.
            timeout: Default command timeout.
        """

        try:
            rospy.init_node("stap_node")
        except rospy.exceptions.ROSException:
            print("Node has already been initialized, do nothing")

        self._gripper_command_pub = rospy.Publisher("/sara_shield/gripper_command", Bool, queue_size=100)
        self._gripper_success_sub = rospy.Subscriber(
            "/sara_shield/gripper_success", Bool, self.gripper_success_callback
        )

        self._status = articulated_body.ControlStatus.UNINITIALIZED
        self._sim_status = articulated_body.ControlStatus.UNINITIALIZED

        super().__init__(
            physics_id=physics_id,
            body_id=body_id,
            T_world_to_ee=T_world_to_ee,
            torque_joints=torque_joints,
            position_joints=position_joints,
            finger_links=finger_links,
            base_link=base_link,
            command_multipliers=command_multipliers,
            finger_contact_normals=finger_contact_normals,
            inertia_kwargs=inertia_kwargs,
            pos_gains=pos_gains,
            pos_threshold=pos_threshold,
            timeout=timeout,
            q_home=q_home,
        )

        for link_id in range(self.dof):
            p.setCollisionFilterGroupMask(self.body_id, link_id, 0, 0, physicsClientId=self.physics_id)

    def gripper_success_callback(self, msg: Bool):
        """ROS callback for gripper success."""
        if msg.data:
            self._status = articulated_body.ControlStatus.VEL_CONVERGED
        else:
            self._status = articulated_body.ControlStatus.TIMEOUT

    def get_joint_state(self, joints: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the position and velocities of the given joints.

        Gets the joint state from the real gripper via Redis and applies it to pybullet.

        Args:
            joints: List of joint ids.
        Returns:
            Joint positions and velocities (q, dq).
        """
        super().get_joint_state(joints)
        # if joints != self.joints:
        #     raise NotImplementedError
        # b_gripper_pos = self._redis.get(self._redis_keys.sensor_pos)
        # if b_gripper_pos is None:
        #     raise RuntimeError("Unable to get Redis key:", self._redis_keys.sensor_pos)
        # gripper_pos = float(b_gripper_pos.decode("utf8")) / 255
        # q = self._command_multipliers * gripper_pos
        # # Update pybullet joints.
        # self.apply_positions(q, joints)
        # return q, np.zeros_like(q)

    def reset_joints(self, q: np.ndarray, joints: List[int]) -> None:
        super().reset_joints(q, joints)

    def apply_torques(self, torques: np.ndarray, joints: Optional[List[int]] = None) -> None:
        super().apply_torques(torques, joints)

    def reset(self) -> bool:
        """Removes any grasp constraint and resets the gripper to the open position."""
        super().reset()
        # self._gripper_state = sim_gripper.GripperState()
        # self.set_grasp(0)
        # while self.update_torques() == articulated_body.ControlStatus.IN_PROGRESS:
        #     continue
        return True

    def is_object_grasped(self, body_id: int) -> bool:
        """Detects whether the given body is grasped.

        A body is considered grasped if the gripper is perfectly closed (sim) or
        not mostly closed (real).

        Args:
            body_id: Body id for which to check the grasp.
        Returns:
            True if the body is grasped.
        """
        return (
            self._status == articulated_body.ControlStatus.POS_CONVERGED
            or self._status == articulated_body.ControlStatus.VEL_CONVERGED
        ) and (
            self._sim_status == articulated_body.ControlStatus.POS_CONVERGED
            or self._sim_status == articulated_body.ControlStatus.VEL_CONVERGED
        )

    def set_grasp(
        self,
        command: float,
        pos_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().set_grasp(command, pos_gains, timeout)
        # self._gripper_state.torque_control = True
        self._status = articulated_body.ControlStatus.IN_PROGRESS
        self._sim_status = articulated_body.ControlStatus.IN_PROGRESS
        self._gripper_command_pub.publish(command > 0.0)

    def update_torques(self) -> articulated_body.ControlStatus:
        """Gets the latest status from the Redis gripper controller.

        Returns:
            Controller status.
        """
        # message = self._redis_sub.get_message()
        # while message is not None:
        #     if message["data"].decode("utf8") == "done":
        #         self._redis_sub.unsubscribe(self._redis_keys.control_pub_status)
        #         break
        #     message = self._redis_sub.get_message()
        # # Update sim.
        # q, dq = self.get_joint_state(self.joints)
        # self.apply_positions(q, self.joints)
        # # Return in progress.
        # if message is None:
        #     return articulated_body.ControlStatus.IN_PROGRESS
        # # TODO: Timeout
        # return articulated_body.ControlStatus.VEL_CONVERGED
        self._sim_status = super().update_torques()
        rospy.sleep(SIMULATION_TIME_STEP)
        # print(f"Sim status: {self._sim_status}, Real status: {self._status}")
        return (
            self._status
            if (
                self._sim_status == articulated_body.ControlStatus.POS_CONVERGED
                or self._sim_status == articulated_body.ControlStatus.VEL_CONVERGED
            )
            else self._sim_status
        )

    def set_state(self, state: Dict[str, Any]) -> None:
        # self.set_grasp(0)
        super().set_state(state)
