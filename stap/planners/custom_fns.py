"""Defines custom evaluation functions to be used in the trajectory evaluation of the planner.

Author: Jakob Thumm
Date: 2024-01-02
"""

from typing import Optional, Tuple

import torch

from stap.envs.base import Primitive
from stap.envs.pybullet.table import object_state
from stap.utils.transformation_utils import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotate_vector_by_axis_angle,
)


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


def get_object_head_length(observation: torch.Tensor, id: int) -> torch.Tensor:
    r"""Returns the head length of the object.

    Args:
        observation [batch_size, state_dim]: Current state.
        id: ID of the object.

    Returns:
        head length of the object [batch_size].
    """
    id_feature = list(object_state.ObjectState.RANGES.keys()).index("head_length")
    return observation[:, id, id_feature]


def get_object_handle_length(observation: torch.Tensor, id: int) -> torch.Tensor:
    r"""Returns the handle length of the object.

    Args:
        observation [batch_size, state_dim]: Current state.
        id: ID of the object.

    Returns:
        handle length of the object [batch_size].
    """
    id_feature = list(object_state.ObjectState.RANGES.keys()).index("handle_length")
    return observation[:, id, id_feature]


def get_object_handle_y(observation: torch.Tensor, id: int) -> torch.Tensor:
    """Returns the handle y position of the object.

    Args:
        observation [batch_size, state_dim]: Current state.
        id: ID of the object.

    Returns:
        handle y position of the object [batch_size].
    """
    id_feature = list(object_state.ObjectState.RANGES.keys()).index("handle_y")
    return observation[:, id, id_feature]


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
    r"""Evaluates the position of the pick primitive.

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
    MIN_VALUE = -1.0
    MAX_VALUE = 2.0
    MAX_DIST = 0.2
    eef_pos, eef_aa = get_eef_pose_in_object_frame(next_state, idx, 0)
    # We want to grab the handle.
    # The handle has its center at [0.5 * handle_length, 0.0, 0.0] in the object frame.
    handle_length = get_object_handle_length(next_state, idx)
    handle_center = torch.zeros_like(eef_pos, device=state.device)
    handle_center[:, 0] = 1.5 * handle_length
    position_value = MIN_VALUE + (MAX_DIST - torch.abs(eef_pos[:, 0] - handle_center[:, 0])) / MAX_DIST * (
        MAX_VALUE - MIN_VALUE
    )
    return position_value


def HandoverPositionFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the position of the handover.

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
    idx_obj = arg_object_ids[0]
    idx_hand = arg_object_ids[1]
    object_position = get_object_position(next_state, idx_obj)
    hand_position = get_object_position(state, idx_hand)
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    POS_RANGE = 1.0
    position_value = MIN_VALUE + (POS_RANGE - torch.norm(object_position - hand_position, dim=1)) / POS_RANGE * (
        MAX_VALUE - MIN_VALUE
    )
    return position_value


def HandoverOrientationFn(
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
    idx_obj = arg_object_ids[0]
    idx_hand = arg_object_ids[1]
    object_position = get_object_position(next_state, idx_obj)
    hand_position = get_object_position(state, idx_hand)
    object_orientation = get_object_orientation(next_state, idx_obj)
    # The head of the screwdriver points in negative x-direction in the object frame.
    x_axis = torch.zeros_like(object_orientation, device=object_orientation.device)
    x_axis[..., 0] = -1.0
    new_direction_vector = rotate_vector_by_axis_angle(x_axis, object_orientation)
    # Hand direction before the handover
    hand_direction = hand_position - object_position
    hand_direction = hand_direction / torch.norm(hand_direction, dim=1, keepdim=True)
    # Calculate great circle distance between the two vectors
    dot_product = torch.sum(new_direction_vector * hand_direction, dim=1)
    angle_difference = torch.acos(torch.clip(dot_product, -1.0, 1.0))
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    ANGLE_RANGE = torch.pi
    orientation_value = MIN_VALUE + (ANGLE_RANGE - angle_difference) / ANGLE_RANGE * (MAX_VALUE - MIN_VALUE)
    assert not torch.any(torch.isnan(orientation_value))
    return orientation_value


CUSTOM_FNS = {
    "HookHandoverOrientationFn": HookHandoverOrientationFn,
    "ScrewdriverPickFn": ScrewdriverPickFn,
    "HandoverPositionFn": HandoverPositionFn,
    "HandoverOrientationFn": HandoverOrientationFn,
}
