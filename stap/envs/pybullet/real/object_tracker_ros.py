import pathlib
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation

from stap.envs.pybullet.real.object_tracker import ObjectTracker
from stap.envs.pybullet.sim import math
from stap.envs.pybullet.table.objects import Object


class ObjectTrackerRos(ObjectTracker):
    def __init__(
        self,
        objects: Dict[str, Object],
        redis_host: str,
        redis_port: int,
        redis_password: str,
        key_namespace: str,
        object_key_prefix: str,
        assets_path: Union[str, pathlib.Path],
        base_transform: np.ndarray,
        alpha: float = 1.0,
    ):
        try:
            rospy.init_node("stap_node")
        except rospy.exceptions.ROSException:
            print("Node has already been initialized, do nothing")
        assert base_transform.shape == (4, 4), "Base transform must be a 4x4 matrix"
        self._base_transform = base_transform
        self._object_key_prefix = object_key_prefix
        self._tracked_objects = dict()
        self._object_transform = dict()
        for object in objects.values():
            self._joint_pos_sub = rospy.Subscriber(
                "/vrpn_client_node/objects/" + object.name + "/pose",
                PoseStamped,
                partial(self.object_callback, object_name=object.name),
            )
            self._tracked_objects[object.name] = object
            self._object_transform[object.name] = None
        self.alpha = alpha

    def __del__(self) -> None:
        pass

    def get_tracked_objects(self, objects: Iterable[Object]) -> List[Object]:
        returned_objects = []
        for object in objects:
            if not self._tracked_objects.__contains__(object.name):
                print("The object '" + object.name + "' is not in the list of tracked objects.")
                continue
            returned_objects.append(self._tracked_objects[object.name])
        return returned_objects

    def update_poses(
        self,
        objects: Optional[Iterable[Object]] = None,
        exclude: Optional[Sequence[Object]] = None,
    ) -> List[Object]:
        returned_objects = []
        if objects is None:
            objects = self._tracked_objects.values()
        excluded_object_names = [object.name for object in exclude] if exclude is not None else None
        for object in objects:
            if excluded_object_names is not None and object.name in excluded_object_names:
                continue
            if not self._tracked_objects.__contains__(object.name):
                print("The object '" + object.name + "' is not in the list of tracked objects.")
                continue
            returned_objects.append(self._tracked_objects[object.name])
        return returned_objects

    def send_poses(self, objects: Optional[Iterable[Object]] = None) -> None:
        pass

    def object_callback(self, msg: PoseStamped, object_name: str):
        object_transform = np.eye(4)
        object_transform[:3, 3] = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        object_transform[:3, :3] = Rotation.from_quat(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        ).as_matrix()
        if self._object_transform[object_name] is None:
            self._object_transform[object_name] = object_transform
        # object_transform_diff = np.clip(
        #     object_transform - self._object_transform[object_name], -self.meas_diff, self.meas_diff
        # )
        # self._object_transform[object_name] = self._object_transform[object_name] + object_transform_diff
        self._object_transform[object_name] = (
            self.alpha * object_transform + (1 - self.alpha) * self._object_transform[object_name]
        )
        object_in_base_frame = self._base_transform @ self._object_transform[object_name]
        pose = math.Pose(
            pos=object_in_base_frame[:3, 3],
            quat=Rotation.from_matrix(object_in_base_frame[:3, :3]).as_quat(),
        )
        self._tracked_objects[object_name].set_pose(pose)
