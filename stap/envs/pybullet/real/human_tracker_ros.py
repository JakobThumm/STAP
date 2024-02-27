from functools import partial
from typing import Sequence

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped

from stap.envs.pybullet.real.object_tracker import ObjectTracker


class HumanTrackerRos(ObjectTracker):
    def __init__(
        self,
        human_joints: Sequence[str],
        base_transform: np.ndarray,
    ):
        try:
            rospy.init_node("stap_node")
        except rospy.exceptions.ROSException:
            print("Node has already been initialized, do nothing")
        self._human_joints = human_joints
        assert base_transform.shape == (4, 4), "Base transform must be a 4x4 matrix"
        self._base_transform = base_transform
        self._human_joint_positions = dict()
        for human_joint in self._human_joints:
            self._joint_pos_sub = rospy.Subscriber(
                "/vrpn_client_node/human/" + human_joint + "/pose",
                PoseStamped,
                partial(self.measurement_callback, object_name=human_joint),
            )
            self._human_joint_positions[human_joint] = np.zeros(3)

    def __del__(self) -> None:
        pass

    def get_joint_positions(self) -> np.ndarray:
        positions = np.array([self._human_joint_positions[human_joint] for human_joint in self._human_joints])
        # Append a 1 to the end of each position to make it a N x 4 matrix
        extended_pos = np.hstack((positions, np.ones((positions.shape[0], 1))))
        # Multiply each pos with the base transform to get the position in the simulation frame
        return (self._base_transform @ extended_pos.T).T[:, :3]

    def measurement_callback(self, msg: PoseStamped, object_name: str):
        self._human_joint_positions[object_name] = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )
