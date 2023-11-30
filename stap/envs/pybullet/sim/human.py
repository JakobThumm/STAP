"""This class defines the human model and its animation.

Author: Jakob Thumm
Date: 29.11.2023
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

from stap.envs.pybullet.sim import body, shapes
from stap.envs.pybullet.sim.math import Pose


class ControlException(Exception):
    """An exception raised due to a control fault (e.g. reaching singularity)."""

    pass


def calc_capsule_pose(
    p1: np.ndarray, p2: np.ndarray, position_offset: np.ndarray, base_orientation: np.ndarray
) -> Pose:
    """Calculate the pose of the body from the measurement.

    Args:
        p1 [3]: Measurement of the first point defining a human capsule.
        p2 [3]: Measurement of the second point defining a human capsule.
        position_offset [3]: Offset of the human body.
        base_orientation [4]: Orientation of the human body as quaternion of the form (x, y, z, w).
    """
    animation_offset_rot = Rotation.from_quat(base_orientation)
    p1 = animation_offset_rot.apply(p1)
    p2 = animation_offset_rot.apply(p2)
    p1 += position_offset
    p2 += position_offset
    p1x = p1[0]
    p1y = p1[1]
    p1z = p1[2]
    p2x = p2[0]
    p2y = p2[1]
    p2z = p2[2]
    v2_x = p2x - p1x
    v2_y = p2y - p1y
    v2_z = p2z - p1z
    norm = math.sqrt(math.pow(v2_x, 2) + math.pow(v2_y, 2) + math.pow(v2_z, 2))
    # POS
    pos = np.array([(p1x + p2x) / 2, (p1y + p2y) / 2, (p1z + p2z) / 2])
    quat = np.array([0, 0, 0, 1])
    if norm > 1e-6:
        # ORIENTATION
        # Rotate z axis vector to direction vector according to
        # https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another/1171995#1171995
        a_x = -v2_y / norm
        a_y = v2_x / norm
        a_z = 0
        a_w = 1 + v2_z / norm
        norm_q = math.sqrt(math.pow(a_w, 2) + math.pow(a_x, 2) + math.pow(a_y, 2) + math.pow(a_z, 2))
        if norm_q > 1e-6:
            w = a_w / norm_q
            x = a_x / norm_q
            y = a_y / norm_q
            z = a_z / norm_q
            # Pybullet is x, y, z, w
            quat = np.array([x, y, z, w])
    # pos += position_offset
    # animation_offset_rot = Rotation.from_quat(base_orientation)
    # final_rot = animation_offset_rot.__mul__(Rotation.from_quat(quat))
    # final_quat = final_rot.as_quat()
    return Pose(pos, quat)


class Human(body.Body):
    def __init__(
        self,
        physics_id: int,
        name: str,
        color: Union[List[float], np.ndarray],
        body_names: List[str],
        body_measurement_ids: Dict[str, Tuple[int, int]],
        body_radii: Dict[str, float],
        body_lengths: Dict[str, float],
    ):
        assert len(body_names) > 0
        for body_name in body_names:
            assert body_name in body_measurement_ids
            assert body_name in body_radii
        self._name = name
        self._body_names = body_names
        self._body_measurement_ids = body_measurement_ids
        self._bodies = {}
        for body_name in body_names:
            shape = shapes.Visual(
                shape=shapes.Capsule(
                    mass=0, color=np.array(color), radius=body_radii[body_name], height=body_lengths[body_name]
                )
            )
            body_id = shapes.create_body(shapes=shape, physics_id=physics_id)
            self._bodies[body_name] = body.Body(physics_id=physics_id, body_id=body_id)
        super().__init__(physics_id=physics_id, body_id=self._bodies[body_names[0]].body_id)
        self.freeze()
        self._animation = np.array([])
        self._animation_info = {}
        self._recording_freq = 0.0
        self._start_time = None
        self._allow_animation = True

    def reset(self, action_skeleton: List, initial_state: Optional[List] = None) -> None:
        for body_name in self._body_names:
            self._bodies[body_name].set_pose(Pose(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])))
        self._start_time = None

    def set_animation(self, animation: np.ndarray, animation_info: Dict[str, Any], recording_freq: float) -> None:
        self._animation = animation
        self._animation_info = animation_info
        self._recording_freq = recording_freq

    def animate(self, time: float) -> None:
        if not self._allow_animation:
            return
        if self._start_time is None:
            self._start_time = time
        if self._animation.size == 0:
            return
        run_time = time - self._start_time
        run_step = int(run_time * self._recording_freq)
        animation_idx = run_step % self._animation.shape[0]

        self._set_poses(
            self._animation[animation_idx],
            self._animation_info["position_offset"],
            self._animation_info["orientation_quat"],
        )

    def _set_poses(self, measurements: np.ndarray, position_offset: np.ndarray, base_orientation: np.ndarray) -> None:
        """Set the poses of the bodies according to the measurements.

        Args:
            measurements [n_bodies, 3]: Measurements of the human body.
            position_offset [3]: Offset of the human body.
            base_orientation [4]: Orientation of the human body as quaternion of the form (x, y, z, w).
        """
        for body_name in self._body_names:
            p1 = measurements[self._body_measurement_ids[body_name][0]]
            p2 = measurements[self._body_measurement_ids[body_name][1]]
            pose = calc_capsule_pose(p1, p2, position_offset, base_orientation)
            self._bodies[body_name].set_pose(pose)

    def disable_animation(self) -> None:
        self._allow_animation = False

    def enable_animation(self) -> None:
        self._allow_animation = True

    @property
    def name(self) -> str:
        return self._name
        return self._name
