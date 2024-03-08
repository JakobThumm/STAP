from typing import List, Optional, Union

import numpy as np
from ctrlutils import eigen
from scipy.spatial.transform import Rotation

from stap.envs.pybullet.sim import math


class ObjectState:
    RANGES = {
        "x": (-0.3, 0.9),
        "y": (-0.5, 0.5),
        "z": (-0.1, 1.0),
        "R11": (-1, 1),
        "R21": (-1, 1),
        "R31": (-1, 1),
        "R12": (-1, 1),
        "R22": (-1, 1),
        "R32": (-1, 1),
        "box_size_x": (0.0, 0.4),
        "box_size_y": (0.0, 0.4),
        "box_size_z": (0.0, 0.2),
        "head_length": (0.0, 0.3),
        "handle_length": (0.0, 0.5),
        "handle_y": (-1.0, 1.0),
    }
    FEATURES = {
        "x": {"dynamic"},
        "y": {"dynamic"},
        "z": {"dynamic"},
        "R11": {"dynamic"},
        "R21": {"dynamic"},
        "R31": {"dynamic"},
        "R12": {"dynamic"},
        "R22": {"dynamic"},
        "R32": {"dynamic"},
        "box_size_x": {"static"},
        "box_size_y": {"static"},
        "box_size_z": {"static"},
        "head_length": {"static"},
        "handle_length": {"static"},
        "handle_y": {"static"},
    }

    @classmethod
    def dynamic_feature_indices(cls) -> List[int]:
        return [i for i, f in enumerate(list(cls.FEATURES.values())) if "dynamic" in f]

    @classmethod
    def static_feature_indices(cls) -> List[int]:
        return [i for i, f in enumerate(list(cls.FEATURES.values())) if "static" in f]

    def __init__(self, vector: Optional[np.ndarray] = None):
        if vector is None:
            vector = np.zeros(len(self.RANGES), dtype=np.float32)
        elif vector.shape[-1] != len(self.RANGES):
            vector = vector.reshape(
                (
                    *vector.shape[:-1],
                    vector.shape[-1] // len(self.RANGES),
                    len(self.RANGES),
                )
            )
        self.vector = vector

    @property
    def pos(self) -> np.ndarray:
        return self.vector[..., :3]

    @pos.setter
    def pos(self, pos: np.ndarray) -> None:
        self.vector[..., :3] = pos

    @property
    def rot_mat(self) -> np.ndarray:
        a_1 = self.vector[..., 3:6]
        a_2 = self.vector[..., 6:9]
        b_1 = a_1 / np.linalg.norm(a_1, axis=-1, keepdims=True)
        u_2 = a_2 - np.sum(a_2 * b_1, axis=-1, keepdims=True) * b_1
        b_2 = u_2 / np.linalg.norm(u_2, axis=-1, keepdims=True)
        b_3 = np.cross(b_1, b_2)
        return np.stack((b_1, b_2, b_3), axis=-1)

    @rot_mat.setter
    def rot_mat(self, rot_mat: np.ndarray) -> None:
        self.vector[..., 3:9] = rot_mat[:, :2].reshape(6, order="F")

    @property
    def aa(self) -> np.ndarray:
        return Rotation.from_matrix(self.rot_mat).as_rotvec()

    @aa.setter
    def aa(self, aa: np.ndarray) -> None:
        self.rot_mat = Rotation.from_rotvec(aa).as_matrix()

    @property
    def box_size(self) -> np.ndarray:
        return self.vector[..., 9:12]

    @box_size.setter
    def box_size(self, box_size: np.ndarray) -> None:
        self.vector[..., 9:12] = box_size

    @property
    def head_length(self) -> Union[float, np.ndarray]:
        if self.vector.ndim > 1:
            return self.vector[..., 12:13]
        return self.vector[12]

    @head_length.setter
    def head_length(self, head_length: Union[float, np.ndarray]) -> None:
        self.vector[..., 12:13] = head_length

    @property
    def handle_length(self) -> Union[float, np.ndarray]:
        if self.vector.ndim > 1:
            return self.vector[..., 13:14]
        return self.vector[13]

    @handle_length.setter
    def handle_length(self, handle_length: Union[float, np.ndarray]) -> None:
        self.vector[..., 13:14] = handle_length

    @property
    def handle_y(self) -> Union[float, np.ndarray]:
        if self.vector.ndim > 1:
            return self.vector[..., 14:15]
        return self.vector[14]

    @handle_y.setter
    def handle_y(self, handle_y: Union[float, np.ndarray]) -> None:
        self.vector[..., 15:16] = handle_y

    @classmethod
    def range(cls) -> np.ndarray:
        return np.array(list(cls.RANGES.values()), dtype=np.float32).T

    def pose(self) -> math.Pose:
        angle = np.linalg.norm(self.aa)
        if angle == 0:
            quat = eigen.Quaterniond.identity()
        else:
            axis = self.aa / angle
            quat = eigen.Quaterniond(eigen.AngleAxisd(angle, axis))
        return math.Pose(pos=self.pos, quat=quat.coeffs)

    def set_pose(self, pose: math.Pose) -> None:
        aa = eigen.AngleAxisd(eigen.Quaterniond(pose.quat))
        self.pos = pose.pos
        self.aa = aa.angle * aa.axis

    def __repr__(self) -> str:
        return (
            "{\n"
            f"    pos: {self.pos},\n"
            f"    aa: {self.aa},\n"
            f"    box_size: {self.box_size},\n"
            f"    head_length: {self.head_length},\n"
            f"    handle_length: {self.handle_length},\n"
            f"    handle_y: {self.handle_y},\n"
            "}"
        )
