"""Defines custom evaluation functions to be used in the trajectory evaluation of the planner.

Author: Jakob Thumm
Date: 2024-01-02
"""

from typing import Optional, Tuple

import torch

from stap.envs.base import Primitive
from stap.envs.pybullet.table import object_state


def get_object_orientation(observation: torch.Tensor, id: int) -> torch.Tensor:
    r"""Returns the orientation of the object in axis-angle representation.

    The returned orientation is in axis-angle representation, where:
        - angle = ||w||_2
        - axis = w / angle

    Args:
        observation [batch_size, state_dim]: Current state.
        id: ID of the object.

    Returns:
        Orientation of the object [batch_size, 3].
    """
    idxwx = list(object_state.ObjectState.RANGES.keys()).index("wx")
    idxwy = list(object_state.ObjectState.RANGES.keys()).index("wy")
    idxwz = list(object_state.ObjectState.RANGES.keys()).index("wz")
    object_orientation = torch.zeros([observation.shape[0], 3], device=observation.device)
    object_orientation[:, 0] = observation[:, id, idxwx]
    object_orientation[:, 1] = observation[:, id, idxwy]
    object_orientation[:, 2] = observation[:, id, idxwz]
    return object_orientation


def get_object_position(observation: torch.Tensor, id: int) -> torch.Tensor:
    r"""Returns the position of the object.

    Args:
        observation [batch_size, state_dim]: Current state.
        id: ID of the object.

    Returns:
        Position of the object [batch_size, 3].
    """
    idxpx = list(object_state.ObjectState.RANGES.keys()).index("x")
    idxpy = list(object_state.ObjectState.RANGES.keys()).index("y")
    idxpz = list(object_state.ObjectState.RANGES.keys()).index("z")
    object_position = torch.zeros([observation.shape[0], 3], device=observation.device)
    object_position[:, 0] = observation[:, id, idxpx]
    object_position[:, 1] = observation[:, id, idxpy]
    object_position[:, 2] = observation[:, id, idxpz]
    return object_position


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert an axis-angle representation to a rotation matrix.

    Args:
        axis_angle [batch_size, 3]: Axis-angle representation.

    Returns:
        Rotation matrices [batch_size, 3, 3].
    """
    batch_size, _ = axis_angle.shape

    # Compute the angle (norm of each axis_angle vector)
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)

    # Safe normalization of the axis
    axis = axis_angle / torch.clamp(angle, min=1e-6)

    # Components for Rodrigues' formula
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    one_minus_cos = 1 - cos_angle

    # Cross-product matrix for each axis
    zero = torch.zeros(batch_size, 1, dtype=axis_angle.dtype, device=axis_angle.device)
    a_x = axis[..., 0:1]
    a_y = axis[..., 1:2]
    a_z = axis[..., 2:3]

    skew_symmetric = torch.cat(
        [
            torch.cat([zero, -a_z, a_y], dim=-1).unsqueeze(-2),
            torch.cat([a_z, zero, -a_x], dim=-1).unsqueeze(-2),
            torch.cat([-a_y, a_x, zero], dim=-1).unsqueeze(-2),
        ],
        dim=-2,
    )

    # Outer product of axis vectors
    outer = axis.unsqueeze(-1) * axis.unsqueeze(-2)

    # Identity matrix
    identity_matrix = (
        torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).unsqueeze(0).repeat(batch_size, 1, 1)
    )

    # Rodrigues' formula
    rotation_matrix = (
        cos_angle.unsqueeze(-1).unsqueeze(-1) * identity_matrix
        + one_minus_cos.unsqueeze(-1).unsqueeze(-1) * outer
        + sin_angle.unsqueeze(-1).unsqueeze(-1) * skew_symmetric
    )

    return rotation_matrix


def matrix_to_axis_angle(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle representation.

    Args:
        rotation_matrices [batch_size, 3, 3]: Rotation matrices.

    Returns:
        Axis-angle representation [batch_size, 3].
    """
    # Compute trace
    trace = rotation_matrices[..., 0, 0] + rotation_matrices[..., 1, 1] + rotation_matrices[..., 2, 2]

    # Compute angle
    angle = torch.acos(torch.clamp((trace - 1) / 2.0, -1.0, 1.0))

    # Compute axis components
    axis_x = rotation_matrices[..., 2, 1] - rotation_matrices[..., 1, 2]
    axis_y = rotation_matrices[..., 0, 2] - rotation_matrices[..., 2, 0]
    axis_z = rotation_matrices[..., 1, 0] - rotation_matrices[..., 0, 1]

    # Stack axis components
    axis_unnormalized = torch.stack([axis_x, axis_y, axis_z], dim=-1)

    # Normalize axis to ensure it's a unit vector
    axis = axis_unnormalized / torch.linalg.norm(axis_unnormalized, dim=-1, keepdim=True).clamp(min=1e-6)

    # Handle the case of zero rotation (angle close to 0)
    axis[angle.unsqueeze(-1).expand_as(axis) < 1e-6] = 0.0

    # Multiply axis by angle to get axis-angle representation
    axis_angle = axis * angle.unsqueeze(-1)

    return axis_angle


def get_eef_pose_in_object_frame(
    observation: torch.Tensor, obj_id: int, eef_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Returns the end-effector pose in the object frame.

    The returned orientation is in axis-angle representation, where:
        - angle = ||w||_2
        - axis = w / angle

    Args:
        observation [batch_size, state_dim]: Current state.
        obj_id: ID of the object.
        eef_id: ID of the end-effector.
    Returns:
        End-effector position in the object frame [batch_size, 3],
        End-effector orientation in the object frame [batch_size, 3].
    """
    object_position = get_object_position(observation, obj_id)
    object_orientation = get_object_orientation(observation, obj_id)
    eef_position = get_object_position(observation, eef_id)
    eef_orientation = get_object_orientation(observation, eef_id)
    # Convert orientations to rotation matrices
    R_object = axis_angle_to_matrix(object_orientation)
    R_eef = axis_angle_to_matrix(eef_orientation)

    # Compute the inverse of the object's rotation matrix
    R_object_inv = R_object.transpose(-2, -1)

    # Calculate the relative position
    relative_position = torch.matmul(R_object_inv, (eef_position - object_position).unsqueeze(-1)).squeeze(-1)

    # Calculate relative orientation
    R_relative = torch.matmul(R_object_inv, R_eef)

    # Assuming a function to convert rotation matrices back to axis-angle format is available
    relative_orientation = matrix_to_axis_angle(R_relative)
    return (relative_position, relative_orientation)


def HookHandoverOrientationFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the orientation of the hook handover.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    arg_object_ids = primitive.get_policy_args_ids()
    idx = arg_object_ids[0]
    object_position = get_object_position(next_state, idx)
    object_orientation = get_object_orientation(next_state, idx)
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    OPTIMAL_ORIENTATION = -torch.pi / 2
    orientation_value = MIN_VALUE + (torch.abs(object_orientation[:, 2] - OPTIMAL_ORIENTATION)) / (2 * torch.pi) * (
        MAX_VALUE - MIN_VALUE
    )
    OPTIMAL_POSITION = torch.tensor([0.0, -0.7, 0.2], device=state.device)
    POS_RANGE = 1.0
    position_value = MIN_VALUE + (POS_RANGE - torch.norm(object_position - OPTIMAL_POSITION, dim=1)) / POS_RANGE * (
        MAX_VALUE - MIN_VALUE
    )
    return_value = torch.clip((orientation_value + position_value) / 2.0, MIN_VALUE, MAX_VALUE)
    return return_value


def ScrewdriverPickFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the orientation of the hook handover.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    arg_object_ids = primitive.get_policy_args_ids()
    idx = arg_object_ids[0]
    object_position = get_object_position(next_state, idx)
    object_orientation = get_object_orientation(next_state, idx)
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    return torch.ones_like(state[:, 0], device=state.device) * MAX_VALUE


def HandoverPositionFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the orientation of the hook handover.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    arg_object_ids = primitive.get_policy_args_ids()
    idx = arg_object_ids[0]
    object_position = get_object_position(next_state, idx)
    object_orientation = get_object_orientation(next_state, idx)
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    return torch.ones_like(state[:, 0], device=state.device) * MAX_VALUE


CUSTOM_FNS = {
    "HookHandoverOrientationFn": HookHandoverOrientationFn,
    "ScrewdriverPickFn": ScrewdriverPickFn,
    "HandoverPositionFn": HandoverPositionFn,
}
