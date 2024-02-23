import pathlib
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped

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
    ):
        try:
            rospy.init_node("stap_node")
        except rospy.exceptions.ROSException:
            print("Node has already been initialized, do nothing")
        self._object_key_prefix = object_key_prefix
        self._tracked_objects = dict()
        for object in objects.values():
            self._joint_pos_sub = rospy.Subscriber("/stap/objects/" + object.name, PoseStamped, self.object_callback)
            self._tracked_objects[object.name] = deepcopy(object)

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
            objects = self._tracked_objects
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

    def object_callback(self, msg: PoseStamped):
        # TODO figure out if that is the right data format.
        pose = math.Pose(np.array(msg.pose.pos), np.array(msg.pose.quat))
        self._tracked_objects[msg.header.name].set_pose(pose)
