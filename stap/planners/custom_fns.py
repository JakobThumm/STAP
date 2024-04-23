"""Defines custom evaluation functions to be used in the trajectory evaluation of the planner.

Author: Jakob Thumm
Date: 2024-01-02
"""

from typing import Optional

import torch

from stap.envs.base import Primitive
from stap.planners.custom_fn_utils import (  # noqa: F401
    build_direction_vector,
    generate_pose_batch,
    get_eef_pose_in_object_frame,
    get_object_handle_length,
    get_object_handle_y,
    get_object_head_length,
    get_object_id_from_name,
    get_object_id_from_primitive,
    get_object_orientation,
    get_object_position,
    get_pose,
    great_circle_distance_metric,
    linear_probability,
    normal_probability,
    pointing_in_direction_metric,
    position_diff_along_direction,
    position_metric_normal_to_direction,
    position_norm_metric,
    probability_intersection,
    probability_union,
    rotation_angle_metric,
    threshold_probability,
)


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

    We want to grasp the screwdirver at the rod, so that the human can easily grab the handle.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    end_effector_id = get_object_id_from_name("end_effector", env, primitive)
    # Get the pose of the end effector in the object frame
    next_end_effector_pose = get_pose(next_state, end_effector_id, object_id)
    # Assumes the rod length is 0.075 and the rod in the positive x direction in object frame.
    preferred_grasp_pose = torch.FloatTensor([0.075 / 2.0, 0, 0, 1, 0, 0, 0]).to(next_state.device)
    # Calculate the positional norm metric
    position_metric = position_norm_metric(next_end_effector_pose, preferred_grasp_pose, norm="L2", axes=["x"])
    # Calculate the probability
    probability_grasp_handle = threshold_probability(position_metric, 0.075 / 2.0, is_smaller_then=True)
    return probability_grasp_handle


def ScrewdriverPickActionFn(
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
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    threshold_greater = 0.2
    position_value_1 = MAX_VALUE * (action[:, 0] > threshold_greater) + MIN_VALUE * (action[:, 0] < threshold_greater)
    threshold_smaller = 0.9
    position_value_2 = MAX_VALUE * (action[:, 0] < threshold_smaller) + MIN_VALUE * (action[:, 0] > threshold_smaller)
    position_value = position_value_1 * position_value_2
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
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    hand_id = get_object_id_from_primitive(1, primitive)
    next_object_pose = get_pose(next_state, object_id)
    current_hand_pose = get_pose(state, hand_id)
    handle_main_axis = [-1.0, 0.0, 0.0]
    # We want to know if the handle is pointing towards the hand position after the handover action.
    orientation_metric = pointing_in_direction_metric(next_object_pose, current_hand_pose, handle_main_axis)
    lower_threshold = torch.pi / 6.0
    upper_threshold = torch.pi / 4.0
    # Calculate the probability
    probability_handover_orientation = linear_probability(
        orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return probability_handover_orientation


def HandoverOrientationAndPositionnFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the orientation and position of the screwdriver handover.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    hand_id = get_object_id_from_primitive(1, primitive)
    next_object_pose = get_pose(next_state, object_id)
    current_hand_pose = get_pose(state, hand_id)
    next_hand_pose = get_pose(next_state, hand_id)
    handle_main_axis = [-1.0, 0.0, 0.0]
    # We want to know if the handle is pointing towards the hand position after the handover action.
    orientation_metric = pointing_in_direction_metric(next_object_pose, current_hand_pose, handle_main_axis)
    lower_threshold = torch.pi / 6.0
    upper_threshold = torch.pi / 4.0
    # Calculate the probability
    probability_handover_orientation = linear_probability(
        orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # We want to be close to the human hand.
    position_metric = position_norm_metric(next_object_pose, next_hand_pose, norm="L2", axes=["x", "y", "z"])
    # Handing over the object an arm length ~0.8m away is considered a failure and close ~0.2m is preferred.
    lower_threshold = 0.2
    upper_threshold = 0.8
    probability_handover_position = linear_probability(
        position_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    total_probability = probability_intersection(probability_handover_position, probability_handover_orientation)
    return total_probability


def HandoverVerticalOrientationFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the if the screwdriver is facing upwards or downwards during handover.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id)
    handle_main_axis = [-1.0, 0.0, 0.0]
    # We want to know if the handle is pointing upwards or downwards after the handover action.
    direction = torch.zeros_like(next_object_pose)
    direction[:, 2] = 1.0
    orientation_metric_up = pointing_in_direction_metric(next_object_pose, direction, handle_main_axis)
    direction[:, 2] = -1.0
    orientation_metric_down = pointing_in_direction_metric(next_object_pose, direction, handle_main_axis)
    lower_threshold = torch.pi / 6.0
    upper_threshold = torch.pi / 4.0
    # Calculate the probability
    probability_handover_up = linear_probability(
        orientation_metric_up, lower_threshold, upper_threshold, is_smaller_then=True
    )
    probability_handover_down = linear_probability(
        orientation_metric_down, lower_threshold, upper_threshold, is_smaller_then=True
    )
    total_probability = probability_union(probability_handover_up, probability_handover_down)
    return total_probability


def PlaceLeftOfRedBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed left of the red box and if the object is placed 10cm next to the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed left of the red box
    left = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, left)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed left of the red box.
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate if the object has a deviation in the x direction.
    x_diff_metric = position_metric_normal_to_direction(next_object_pose, red_box_pose, left)
    lower_threshold = 0.0
    upper_threshold = 0.05
    # The x difference should be as small as possible but no larger than 5cm.
    x_diff_probability = linear_probability(x_diff_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    total_left_probability = probability_intersection(is_left_probability, x_diff_probability)
    # Evaluate if the object is placed 10cm next to the red box.
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.050
    ideal_point = 0.10
    upper_threshold = 0.20
    smaller_than_ideal_probability = linear_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability(
        distance_metric, ideal_point, upper_threshold, is_smaller_then=True
    )
    total_distance_probability = probability_intersection(smaller_than_ideal_probability, bigger_than_ideal_probability)
    # Combine the two probabilities
    total_probability = probability_intersection(total_left_probability, total_distance_probability)
    return total_probability


def PlaceInFrontOfBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed in front of the blue box and if the object is placed 10cm next to the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed in front of the blue box
    in_front_of = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, in_front_of)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed left of the blue box.
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate if the object has a deviation in the x direction.
    y_diff_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, in_front_of)
    lower_threshold = 0.0
    upper_threshold = 0.05
    # The x difference should be as small as possible but no larger than 5cm.
    x_diff_probability = linear_probability(y_diff_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    total_left_probability = probability_intersection(is_left_probability, x_diff_probability)
    # Evaluate if the object is placed 10cm next to the blue box.
    distance_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.050
    ideal_point = 0.10
    upper_threshold = 0.20
    smaller_than_ideal_probability = linear_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability(
        distance_metric, ideal_point, upper_threshold, is_smaller_then=True
    )
    total_distance_probability = probability_intersection(smaller_than_ideal_probability, bigger_than_ideal_probability)
    # Combine the two probabilities
    total_probability = probability_intersection(total_left_probability, total_distance_probability)
    return total_probability


def PlaceDistanceApartBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed 20cm next to the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed 20cm next to the blue box.
    distance_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.015
    ideal_point = 0.20
    upper_threshold = 0.25
    smaller_than_ideal_probability = linear_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability(
        distance_metric, ideal_point, upper_threshold, is_smaller_then=True
    )
    total_distance_probability = probability_intersection(smaller_than_ideal_probability, bigger_than_ideal_probability)
    return total_distance_probability


##### Hand-written custom functions for object arrangement ablation study #####
# Place the object in a line with the red box and the blue box.
def PlaceInLineWithRedAndBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed in line with the red and blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    red_to_blue = build_direction_vector(red_box_pose, blue_box_pose)
    normal_distance_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, red_to_blue)
    lower_threshold = 0.0
    upper_threshold = 0.05
    # The x difference should be as small as possible but no larger than 5cm.
    probability = linear_probability(normal_distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return probability


def PlaceNextToBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed next to the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed at least 10cm next to the blue box.
    distance_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10
    upper_threshold = 0.15
    close_by_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return close_by_probability


def PlaceFarAwayFromRedAndBlueFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed far away from the red and blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed far away from the red box
    distance_metric_red = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.20
    upper_threshold = 1.0
    far_away_probability_red = linear_probability(
        distance_metric_red, lower_threshold, upper_threshold, is_smaller_then=False
    )
    # Evaluate if the object is placed far away from the blue box
    distance_metric_blue = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    far_away_probability_blue = linear_probability(
        distance_metric_blue, lower_threshold, upper_threshold, is_smaller_then=False
    )
    # Combine the two probabilities
    total_probability = probability_intersection(far_away_probability_red, far_away_probability_blue)
    return total_probability


def PlaceNextToRedBox20cmFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed 20cm next to the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed 20cm next to the red box.
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.15
    ideal_point = 0.20
    upper_threshold = 0.25
    smaller_than_ideal_probability = linear_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability(
        distance_metric, ideal_point, upper_threshold, is_smaller_then=True
    )
    return probability_intersection(smaller_than_ideal_probability, bigger_than_ideal_probability)


def PlaceNextToRedBoxAndBlueBox20cmFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed 20cm next to the red and blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed 20cm next to the red box.
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.15
    ideal_point = 0.20
    upper_threshold = 0.25
    smaller_than_ideal_probability = linear_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability(
        distance_metric, ideal_point, upper_threshold, is_smaller_then=True
    )
    probability_red_box = probability_intersection(smaller_than_ideal_probability, bigger_than_ideal_probability)
    distance_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    smaller_than_ideal_probability = linear_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability(
        distance_metric, ideal_point, upper_threshold, is_smaller_then=True
    )
    probability_blue_box = probability_intersection(smaller_than_ideal_probability, bigger_than_ideal_probability)
    return probability_intersection(probability_red_box, probability_blue_box)


def LeftOfRedBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed left of the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed left of the red box
    left = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, left)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed left of the red box.
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_left_probability


def BehindBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed behind the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed behind the blue box
    behind = [-1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, behind)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed behind the blue box.
    is_behind_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_behind_probability


def InFrontOfBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed in front the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed in front the blue box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, in_front)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed in front the blue box.
    is_in_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_in_front_probability


def PlaceNextToScrewdriver15cmFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed 15cm next to the screwdriver.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    screwdriver_id = get_object_id_from_name("screwdriver", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    screwdriver_pose = get_pose(state, screwdriver_id, -1)
    # Evaluate if the object is placed 15cm next to the screwdriver.
    distance_metric = position_norm_metric(next_object_pose, screwdriver_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10
    ideal_point = 0.15
    upper_threshold = 0.20
    smaller_than_ideal_probability = linear_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability(
        distance_metric, ideal_point, upper_threshold, is_smaller_then=True
    )
    return probability_intersection(smaller_than_ideal_probability, bigger_than_ideal_probability)


def FarLeftOfTableFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed far left of the table.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    table_origin = generate_pose_batch([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], next_object_pose)
    # Evaluate if the object is placed far left of the table
    left = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, table_origin, left)
    mean = 0.5
    std = 0.2
    is_left_probability = normal_probability(direction_difference, mean, std, is_smaller_then=False)
    return is_left_probability


def CloseToCyanBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed close to the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the object is placed close to the cyan box
    distance_metric = position_norm_metric(next_object_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10
    upper_threshold = 0.15
    closeness_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return closeness_probability


def CloseToCyanAndBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed close to the cyan and blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed close to the cyan box
    distance_metric = position_norm_metric(next_object_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10
    upper_threshold = 0.15
    closeness_probability_cyan = linear_probability(
        distance_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Evaluate if the object is placed close to the blue box
    distance_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    closeness_probability_blue = linear_probability(
        distance_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return probability_intersection(closeness_probability_cyan, closeness_probability_blue)


def SameOrientationAsCyanBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object has the same orientation as the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the object has the same orientation as the cyan box
    orientation_metric = great_circle_distance_metric(next_object_pose, cyan_box_pose)
    lower_threshold = torch.pi / 6.0
    upper_threshold = torch.pi / 3.0
    orientation_probability = linear_probability(
        orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return orientation_probability


def InFrontOfCyanBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed in front of the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the object is placed in front of the cyan box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, cyan_box_pose, in_front)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed in front of the cyan box.
    is_in_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_in_front_probability


def AlignBoxesInLineFn_trial_0(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in line with the red and blue boxes.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate the direction vector from the red box to the blue box
    direction_vector = build_direction_vector(red_box_pose, blue_box_pose)
    # Calculate the positional difference of the cyan box normal to the direction vector
    normal_difference_metric = position_metric_normal_to_direction(cyan_box_pose, red_box_pose, direction_vector)
    # The cyan box should be aligned with the red and blue boxes, so the normal difference should be minimal
    alignment_threshold = 0.05  # Allowable deviation in meters
    alignment_probability = threshold_probability(normal_difference_metric, alignment_threshold, is_smaller_then=True)
    return alignment_probability


def NextToBlueBoxFn_trial_1(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the red box is placed next to the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    red_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_red_box_pose = get_pose(next_state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed next to the blue box
    distance_metric = position_norm_metric(next_red_box_pose, blue_box_pose, norm="L2")
    # Assuming a threshold for "next to" as 0.15 meters
    is_next_to_probability = threshold_probability(distance_metric, 0.15, is_smaller_then=True)
    return is_next_to_probability


def FarFromBlueAndRedBoxFn_trial_1(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed far from both the blue and red boxes.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate distance from both boxes
    distance_to_blue = position_norm_metric(next_cyan_box_pose, blue_box_pose, norm="L2")
    distance_to_red = position_norm_metric(next_cyan_box_pose, red_box_pose, norm="L2")
    # Assuming a threshold for "far from" as 0.5 meters
    is_far_from_blue_probability = threshold_probability(distance_to_blue, 0.5, is_smaller_then=False)
    is_far_from_red_probability = threshold_probability(distance_to_red, 0.5, is_smaller_then=False)
    # Combine probabilities to ensure it's far from both
    total_probability = probability_intersection(is_far_from_blue_probability, is_far_from_red_probability)
    return total_probability


def TrianglePlacementBlueFn_trial_2(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in a triangle formation with the red box at a distance of 20cm.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the blue box is placed 20cm away from the red box
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    ideal_distance = 0.20  # 20cm
    distance_probability = threshold_probability(distance_metric, ideal_distance, is_smaller_then=False)
    return distance_probability


def TrianglePlacementCyanFn_trial_2(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in a triangle formation with the red and blue boxes at a distance of 20cm.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the cyan box is placed 20cm away from both the red and blue boxes
    distance_metric_red = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_blue = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    ideal_distance = 0.20  # 20cm
    distance_probability_red = threshold_probability(distance_metric_red, ideal_distance, is_smaller_then=False)
    distance_probability_blue = threshold_probability(distance_metric_blue, ideal_distance, is_smaller_then=False)
    total_probability = probability_intersection(distance_probability_red, distance_probability_blue)
    return total_probability


def LeftOfRedBoxFn_trial_3(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed to the left of the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed to the left of the red box
    left_of = [0.0, 1.0, 0.0]  # Assuming positive y-direction is "left"
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, left_of)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed to the left of the red box.
    is_left_of_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_left_of_probability


def PlaceBlueInFrontOfRedFn_trial_4(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in front of the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the blue box is placed in front of the red box
    front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(blue_box_pose, red_box_pose, front)
    lower_threshold = 0.0
    # The direction difference should be positive if the blue box is placed in front of the red box.
    in_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return in_front_probability


def PlaceBlueBehindCyanFn_trial_4(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed behind the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed behind the cyan box
    behind = [-1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(blue_box_pose, cyan_box_pose, behind)
    lower_threshold = 0.0
    # The direction difference should be positive if the blue box is placed behind the cyan box.
    behind_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return behind_probability


def PlaceInCircleAroundScrewdriver15cmFn_trial_5(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed in a circle of radius 15cm around the screwdriver.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    screwdriver_id = get_object_id_from_name("screwdriver", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    screwdriver_pose = get_pose(state, screwdriver_id, -1)
    # Evaluate if the object is placed in a circle of radius 15cm around the screwdriver.
    distance_metric = position_norm_metric(next_object_pose, screwdriver_pose, norm="L2", axes=["x", "y"])
    ideal_radius = 0.15  # 15cm
    lower_threshold = 0.14  # Slightly smaller than the ideal radius to allow for some tolerance
    upper_threshold = 0.16  # Slightly larger than the ideal radius to allow for some tolerance
    probability = linear_probability(
        distance_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return probability


def FarLeftOnTableFn_trial_6(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed as far to the left of the table as possible.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner. 
            Output shape: [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    table_id = get_object_id_from_name("table", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    table_pose = get_pose(state, table_id, -1)
    # Assuming the left of the table is negative x direction
    left_most_position = table_pose[:, 0] - 1.5  # Half the table width to the left
    distance_to_left_most = position_diff_along_direction(next_object_pose, generate_pose_batch([left_most_position, 0, 0, 1, 0, 0, 0], next_object_pose), [1, 0, 0])
    # The closer to 0, the better, but not exceeding the table's left boundary
    probability = threshold_probability(distance_to_left_most, 0.05, is_smaller_then=True)
    return probability


def PlaceCloseToCyanBoxFn_trial_7(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed close to the cyan box without them touching.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the object is placed close to the cyan box but not touching.
    distance_metric = position_norm_metric(next_object_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.05  # Minimum distance to not touch
    upper_threshold = 0.15  # Maximum distance to still be considered close
    close_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return close_probability


def PlaceCloseToBlueBoxFn_trial_7(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed close to the blue box without them touching.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed close to the blue box but not touching.
    distance_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.05  # Minimum distance to not touch
    upper_threshold = 0.15  # Maximum distance to still be considered close
    close_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return close_probability


def OrientBlueBoxLikeCyanBoxFn_trial_8(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is oriented in the same direction as the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner. 
            Output shape: [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Calculate the orientation difference between the blue box and the cyan box
    orientation_difference = great_circle_distance_metric(blue_box_pose, cyan_box_pose)
    # Define a threshold for the orientation difference (in radians)
    threshold = 0.1  # Small threshold to ensure they are oriented similarly
    return threshold_probability(orientation_difference, threshold, is_smaller_then=True)


def OrientRedBoxLikeCyanBoxFn_trial_8(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is oriented in the same direction as the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner. 
            Output shape: [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    red_box_pose = get_pose(next_state, red_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Calculate the orientation difference between the red box and the cyan box
    orientation_difference = great_circle_distance_metric(red_box_pose, cyan_box_pose)
    # Define a threshold for the orientation difference (in radians)
    threshold = 0.1  # Small threshold to ensure they are oriented similarly
    return threshold_probability(orientation_difference, threshold, is_smaller_then=True)


def PlaceInLineFn_trial_9(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed in a line with the specified order: red in front of blue in front of cyan.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    
    # Direction vector for "in front of" relation
    in_front = [1.0, 0.0, 0.0]
    
    # Evaluate if the blue box is placed in front of the cyan box
    if object_id == blue_box_id:
        target_pose = cyan_box_pose
    # Evaluate if the red box is placed in front of the blue box
    elif object_id == red_box_id:
        target_pose = blue_box_pose
    else:
        # This function should not be called for the cyan box in this context
        raise ValueError("PlaceInLineFn_trial_9 called with incorrect object.")
    
    direction_difference = position_diff_along_direction(next_object_pose, target_pose, in_front)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed in front of the target.
    is_in_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    
    # Evaluate if the objects are placed close to each other in the specified line
    distance_metric = position_norm_metric(next_object_pose, target_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10
    upper_threshold = 0.20
    closeness_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    
    # Combine the probabilities
    total_probability = probability_intersection(is_in_front_probability, closeness_probability)
    return total_probability


def LeftAndAlignedWithRedBoxFn_trial_10(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed left of and aligned with the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the cyan box is placed left of the red box
    left = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(next_cyan_box_pose, red_box_pose, left)
    left_probability = threshold_probability(direction_difference, 0.0, is_smaller_then=False)
    # Evaluate if the cyan box is aligned with the red box
    alignment_metric = position_metric_normal_to_direction(next_cyan_box_pose, red_box_pose, left)
    alignment_probability = threshold_probability(alignment_metric, 0.05, is_smaller_then=True)
    # Combine probabilities
    total_probability = probability_intersection(left_probability, alignment_probability)
    return total_probability


def PlaceInFrontAndAlignWithBlueBoxFn_trial_11(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in front of the blue box and aligned with it.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed in front of the blue box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, in_front)
    front_threshold = 0.0
    is_in_front_probability = threshold_probability(direction_difference, front_threshold, is_smaller_then=False)
    # Evaluate alignment with the blue box
    normal_distance_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, in_front)
    alignment_threshold = 0.05  # Allow a small margin for alignment
    alignment_probability = threshold_probability(normal_distance_metric, alignment_threshold, is_smaller_then=True)
    # Combine the probabilities
    total_probability = probability_intersection(is_in_front_probability, alignment_probability)
    return total_probability


def LeftAndAlignedWithRedBoxFn_trial_12(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the blue box is placed left of the red box and aligned with it.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_blue_box_pose = get_pose(next_state, blue_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the blue box is placed left of the red box
    left_direction = [0.0, -1.0, 0.0]
    left_difference = position_diff_along_direction(next_blue_box_pose, red_box_pose, left_direction)
    left_probability = threshold_probability(left_difference, 0.0, is_smaller_then=False)
    # Evaluate if the blue box is aligned with the red box
    alignment_difference = position_metric_normal_to_direction(next_blue_box_pose, red_box_pose, left_direction)
    alignment_probability = threshold_probability(alignment_difference, 0.05, is_smaller_then=True)
    # Combine probabilities
    total_probability = probability_intersection(left_probability, alignment_probability)
    return total_probability


def AlignBoxesInLineFn_trial_13(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed in line with the red and blue boxes.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)

    # Calculate the direction vector from the red box to the blue box
    direction_red_to_blue = build_direction_vector(red_box_pose, blue_box_pose)
    # Calculate the normal distance of the cyan box to the line formed by red and blue boxes
    distance_to_line = position_metric_normal_to_direction(cyan_box_pose, red_box_pose, direction_red_to_blue)

    # Define thresholds for alignment
    lower_threshold = 0.0
    upper_threshold = 0.05  # Allow a small margin for alignment

    # Calculate probability based on the distance to the line
    alignment_probability = linear_probability(
        distance_to_line, lower_threshold, upper_threshold, is_smaller_then=True
    )

    return alignment_probability


def PlaceNextToBlueBoxFn_trial_14(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed next to the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed next to the blue box.
    distance_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.05
    ideal_point = 0.10
    upper_threshold = 0.20
    smaller_than_ideal_probability = linear_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability(
        distance_metric, ideal_point, upper_threshold, is_smaller_then=True
    )
    return probability_intersection(smaller_than_ideal_probability, bigger_than_ideal_probability)


def PlaceFarFromBlueAndRedBoxFn_trial_14(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed as far away as possible from both the blue and red boxes.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate distance from blue box
    distance_to_blue = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    # Evaluate distance from red box
    distance_to_red = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    # Combine distances to evaluate farthest possible placement
    combined_distance = distance_to_blue + distance_to_red
    lower_threshold = 0.40  # Combined distance should be more than 40cm
    upper_threshold = 0.60  # Up to 60cm considered optimal
    probability = linear_probability(
        combined_distance, lower_threshold, upper_threshold, is_smaller_then=False
    )
    return probability


def TrianglePlacementBlueFn_trial_15(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in a triangle formation with the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed 20cm from the red box
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    ideal_distance = 0.20  # 20cm
    lower_threshold = 0.18  # Slightly less to allow for some margin
    upper_threshold = 0.22  # Slightly more to allow for some margin
    distance_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return distance_probability


def TrianglePlacementCyanFn_trial_15(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in a triangle formation with the red and blue boxes.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed 20cm from both the red and blue boxes
    distance_metric_red = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_blue = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    ideal_distance = 0.20  # 20cm
    lower_threshold = 0.18  # Slightly less to allow for some margin
    upper_threshold = 0.22  # Slightly more to allow for some margin
    distance_probability_red = linear_probability(distance_metric_red, lower_threshold, upper_threshold, is_smaller_then=True)
    distance_probability_blue = linear_probability(distance_metric_blue, lower_threshold, upper_threshold, is_smaller_then=True)
    # Combine the probabilities to ensure the object is correctly placed in relation to both boxes
    total_probability = probability_intersection(distance_probability_red, distance_probability_blue)
    return total_probability


def LeftOfRedBoxFn_trial_16(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed left of the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed left of the red box
    left_of = [0.0, 1.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, left_of)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed left of the red box.
    is_left_of_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_left_of_probability


def InFrontOfRedBoxBehindCyanBoxFn_trial_17(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in front of the red box but behind the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed in front of the red box
    in_front_of_red = position_diff_along_direction(blue_box_pose, red_box_pose, [1.0, 0.0, 0.0])
    in_front_of_red_probability = threshold_probability(in_front_of_red, 0.0, is_smaller_then=False)
    # Evaluate if the blue box is placed behind the cyan box
    behind_cyan = position_diff_along_direction(cyan_box_pose, blue_box_pose, [1.0, 0.0, 0.0])
    behind_cyan_probability = threshold_probability(behind_cyan, 0.0, is_smaller_then=False)
    # Combine the probabilities
    total_probability = probability_intersection(in_front_of_red_probability, behind_cyan_probability)
    return total_probability


def CircleAroundScrewdriverFn_trial_18(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed in a circle of radius 15 cm around the screwdriver.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    screwdriver_id = get_object_id_from_name("screwdriver", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    screwdriver_pose = get_pose(state, screwdriver_id, -1)
    # Evaluate if the object is placed in a circle of radius 15 cm around the screwdriver
    distance_metric = position_norm_metric(next_object_pose, screwdriver_pose, norm="L2", axes=["x", "y"])
    radius = 0.15  # 15 cm in meters
    tolerance = 0.02  # 2 cm tolerance for placement
    lower_threshold = radius - tolerance
    upper_threshold = radius + tolerance
    circle_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return circle_probability


def PlaceToLeftFn_trial_19(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed as far to the left of the table as possible.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    table_id = get_object_id_from_name("table", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    table_pose = get_pose(state, table_id, -1)
    # Assuming the table's left edge is at x = -1.5 (half of table's width from the center)
    table_left_edge = generate_pose_batch([-1.5, 0.0, 0.0, 1, 0, 0, 0], next_object_pose)
    # Evaluate how close the object is to the left edge of the table
    distance_to_left_edge = position_diff_along_direction(next_object_pose, table_left_edge, [1, 0, 0])
    # The closer to 0, the better. We assume a threshold of 0.05m as acceptable for being "as far left as possible"
    probability = threshold_probability(distance_to_left_edge, 0.05, is_smaller_then=True)
    return probability


def PlaceCloseToCyanBoxFn_trial_20(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the blue box is placed close to the cyan box without touching.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    next_blue_box_pose = get_pose(next_state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed close to the cyan box without touching.
    distance_metric = position_norm_metric(next_blue_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.05  # Minimum distance to avoid touching
    upper_threshold = 0.10  # Maximum distance to be considered close
    close_without_touching_probability = linear_probability(
        distance_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return close_without_touching_probability


def PlaceCloseToBlueAndCyanBoxFn_trial_20(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the red box is placed close to both the blue and cyan box without touching.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    red_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    next_red_box_pose = get_pose(next_state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the red box is placed close to the blue box without touching.
    distance_metric_blue = position_norm_metric(next_red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_cyan = position_norm_metric(next_red_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.05  # Minimum distance to avoid touching
    upper_threshold = 0.10  # Maximum distance to be considered close
    close_to_blue_probability = linear_probability(
        distance_metric_blue, lower_threshold, upper_threshold, is_smaller_then=True
    )
    close_to_cyan_probability = linear_probability(
        distance_metric_cyan, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine the probabilities to ensure the red box is close to both the blue and cyan box without touching.
    total_probability = probability_intersection(close_to_blue_probability, close_to_cyan_probability)
    return total_probability


def OrientWithCyanBoxFn_trial_21(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is oriented in the same direction as the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    next_object_orientation = get_pose(next_state, object_id, -1)[..., 3:]
    cyan_box_orientation = get_pose(state, cyan_box_id, -1)[..., 3:]
    # Calculate the difference in orientation using the great circle distance metric
    orientation_difference = great_circle_distance_metric(next_object_orientation, cyan_box_orientation)
    # Define a threshold for acceptable orientation difference
    acceptable_threshold = 0.1  # radians
    # Calculate the probability that the orientation difference is within the acceptable threshold
    probability = threshold_probability(orientation_difference, acceptable_threshold, is_smaller_then=True)
    return probability


def PlaceInFrontOfCyanBoxFn_trial_22(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in front of the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    next_blue_box_pose = get_pose(next_state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed in front of the cyan box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_blue_box_pose, cyan_box_pose, in_front)
    lower_threshold = 0.0
    # The direction difference should be positive if the blue box is placed in front of the cyan box.
    is_in_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_in_front_probability


def PlaceInFrontOfBlueBoxFn_trial_22(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in front of the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    red_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_red_box_pose = get_pose(next_state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed in front of the blue box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_red_box_pose, blue_box_pose, in_front)
    lower_threshold = 0.0
    # The direction difference should be positive if the red box is placed in front of the blue box.
    is_in_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_in_front_probability


def LeftAndAlignedWithRedBoxFn_trial_23(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed left of and aligned with the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: Optional primitive to receive the object information from
    Returns:
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner. 
            Output shape: [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    
    # Evaluate if the cyan box is placed left of the red box
    left_direction = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(next_cyan_box_pose, red_box_pose, left_direction)
    is_left_probability = threshold_probability(direction_difference, 0.0, is_smaller_then=False)
    
    # Evaluate if the cyan box is aligned with the red box
    alignment_difference = position_metric_normal_to_direction(next_cyan_box_pose, red_box_pose, left_direction)
    alignment_probability = threshold_probability(alignment_difference, 0.05, is_smaller_then=True)
    
    # Combine probabilities
    total_probability = probability_intersection(is_left_probability, alignment_probability)
    return total_probability


def PlaceInFrontOfBlueBoxFn_trial_24(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed in front of the blue box and if the object is aligned with the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed in front of the blue box
    front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, front)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed in front of the blue box.
    is_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate if the object has a deviation in the y direction.
    y_diff_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, front)
    lower_threshold = 0.0
    upper_threshold = 0.05
    # The y difference should be as small as possible but no larger than 5cm.
    y_diff_probability = linear_probability(y_diff_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    total_front_probability = probability_intersection(is_front_probability, y_diff_probability)
    return total_front_probability


def PlaceLeftOfRedBoxAlignedFn_trial_25(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed left of the red box and if the two boxes are aligned.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed left of the red box
    left = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, left)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed left of the red box.
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate if the object is aligned with the red box in the y-axis.
    y_diff_metric = position_metric_normal_to_direction(next_object_pose, red_box_pose, left)
    alignment_threshold = 0.01  # Allow a small margin for alignment
    alignment_probability = threshold_probability(y_diff_metric, alignment_threshold, is_smaller_then=True)
    # Combine the probabilities for being left and aligned
    total_probability = probability_intersection(is_left_probability, alignment_probability)
    return total_probability


def PlaceInLineWithRedAndBlueFn_trial_26(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in a line with the red and blue boxes.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_next_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)

    # Calculate the direction vector from the red box to the blue box
    direction_vector_red_to_blue = build_direction_vector(red_box_pose, blue_box_pose)
    # Calculate the positional difference of the cyan box along the direction vector
    position_diff_cyan_along_direction = position_diff_along_direction(cyan_box_next_pose, red_box_pose, direction_vector_red_to_blue)
    # Calculate the positional difference of the cyan box normal to the direction vector
    position_diff_cyan_normal_to_direction = position_metric_normal_to_direction(cyan_box_next_pose, red_box_pose, direction_vector_red_to_blue)

    # The cyan box should be along the line formed by the red and blue boxes, so the normal difference should be minimal
    normal_diff_threshold = 0.05  # Allow a small margin for alignment
    normal_diff_probability = threshold_probability(position_diff_cyan_normal_to_direction, normal_diff_threshold, is_smaller_then=True)

    # The cyan box should be between the red and blue boxes along the direction vector, so the positional difference should be positive but less than the distance between red and blue
    distance_red_to_blue = position_norm_metric(red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    between_boxes_probability = (position_diff_cyan_along_direction >= 0) & (position_diff_cyan_along_direction <= distance_red_to_blue)

    # Combine probabilities
    total_probability = probability_intersection(normal_diff_probability, between_boxes_probability)
    return total_probability


def PlaceRedBoxNextToBlueBoxFn_trial_27(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the red box is placed next to the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    red_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_red_box_pose = get_pose(next_state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed next to the blue box.
    distance_metric = position_norm_metric(next_red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.05
    upper_threshold = 0.15
    closeness_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return closeness_probability


def PlaceCyanBoxFarFromRedAndBlueBoxFn_trial_27(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed as far away as possible from the red and blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the cyan box is placed as far away as possible from the red and blue box.
    distance_metric_red = position_norm_metric(next_cyan_box_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_blue = position_norm_metric(next_cyan_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.30  # Minimum distance to consider "far"
    far_probability_red = threshold_probability(distance_metric_red, lower_threshold, is_smaller_then=False)
    far_probability_blue = threshold_probability(distance_metric_blue, lower_threshold, is_smaller_then=False)
    return probability_intersection(far_probability_red, far_probability_blue)


def PlaceInTriangleWithRedAndCyanBoxFn_trial_28(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in a triangle with the red and cyan box with an edge length of 20 cm.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Calculate distances
    red_to_blue_distance = position_norm_metric(red_box_pose, blue_box_pose, norm="L2")
    cyan_to_blue_distance = position_norm_metric(cyan_box_pose, blue_box_pose, norm="L2")
    # Ideal distance is 20 cm
    ideal_distance = 0.20
    lower_threshold = 0.15
    upper_threshold = 0.25
    red_to_blue_probability = linear_probability(
        red_to_blue_distance, lower_threshold, upper_threshold, is_smaller_then=True
    )
    cyan_to_blue_probability = linear_probability(
        cyan_to_blue_distance, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine probabilities
    total_probability = probability_intersection(red_to_blue_probability, cyan_to_blue_probability)
    return total_probability


def PlaceInTriangleWithRedAndBlueBoxFn_trial_28(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in a triangle with the red and blue box with an edge length of 20 cm.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate distances
    red_to_cyan_distance = position_norm_metric(red_box_pose, cyan_box_pose, norm="L2")
    blue_to_cyan_distance = position_norm_metric(blue_box_pose, cyan_box_pose, norm="L2")
    # Ideal distance is 20 cm
    ideal_distance = 0.20
    lower_threshold = 0.15
    upper_threshold = 0.25
    red_to_cyan_probability = linear_probability(
        red_to_cyan_distance, lower_threshold, upper_threshold, is_smaller_then=True
    )
    blue_to_cyan_probability = linear_probability(
        blue_to_cyan_distance, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine probabilities
    total_probability = probability_intersection(red_to_cyan_probability, blue_to_cyan_probability)
    return total_probability


def LeftOfRedBoxFn_trial_29(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed left of the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed left of the red box
    left = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, left)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed left of the red box.
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_left_probability


def InFrontOfRedBoxFn_trial_30(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the red box is placed in front of the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    red_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_red_box_pose = get_pose(next_state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed in front of the blue box
    direction_difference = position_diff_along_direction(next_red_box_pose, blue_box_pose, [1, 0, 0])
    mean = 0.0
    std = 0.1
    in_front_probability = normal_probability(direction_difference, mean, std, is_smaller_then=True)
    return in_front_probability


def BehindBlueBoxFn_trial_30(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed behind the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the cyan box is placed behind the blue box
    direction_difference = position_diff_along_direction(next_cyan_box_pose, blue_box_pose, [-1, 0, 0])
    mean = 0.0
    std = 0.1
    behind_probability = normal_probability(direction_difference, mean, std, is_smaller_then=True)
    return behind_probability


def CircleAroundScrewdriverFn_trial_31(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed in a circle of radius 15 cm around the screwdriver.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    screwdriver_id = get_object_id_from_name("screwdriver", env, primitive)
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    screwdriver_pose = get_pose(state, screwdriver_id, -1)
    # Evaluate if the object is placed in a circle of radius 15 cm around the screwdriver
    distance_metric = position_norm_metric(next_object_pose, screwdriver_pose, norm="L2", axes=["x", "y"])
    radius = 0.15  # 15 cm
    tolerance = 0.02  # 2 cm tolerance for placement
    lower_threshold = radius - tolerance
    upper_threshold = radius + tolerance
    circle_probability = linear_probability(
        distance_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return circle_probability


def FarLeftOfTableFn_trial_32(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed as far to the left of the table as possible.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    table_id = get_object_id_from_name("table", env, primitive)
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    table_pose = get_pose(state, table_id, -1)
    # Evaluate if the object is placed as far to the left of the table as possible
    # Assuming the table's left edge is at x = 0 (table center at x = 1.5 given its width)
    left_edge_x = 0.0
    left_edge_pose = generate_pose_batch([left_edge_x, 0, 0, 1, 0, 0, 0], next_object_pose)
    distance_metric = position_norm_metric(next_object_pose, left_edge_pose, norm="L2", axes=["x"])
    lower_threshold = 0.0
    upper_threshold = 0.15  # Allow a small margin for practical reasons
    far_left_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return far_left_probability


def PlaceCloseToCyanBoxFn_trial_33(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed close to the cyan box without touching.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    next_blue_box_pose = get_pose(next_state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed close to the cyan box.
    distance_metric = position_norm_metric(next_blue_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.05  # Minimum distance to avoid touching
    upper_threshold = 0.10  # Maximum distance to be considered close
    close_by_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return close_by_probability


def PlaceCloseToBlueAndCyanBoxFn_trial_33(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed close to both the blue and cyan boxes without touching.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    red_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    next_red_box_pose = get_pose(next_state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the red box is placed close to the blue box.
    distance_metric_blue = position_norm_metric(next_red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold_blue = 0.05
    upper_threshold_blue = 0.10
    close_by_probability_blue = linear_probability(distance_metric_blue, lower_threshold_blue, upper_threshold_blue, is_smaller_then=True)
    # Evaluate if the red box is placed close to the cyan box.
    distance_metric_cyan = position_norm_metric(next_red_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold_cyan = 0.05
    upper_threshold_cyan = 0.10
    close_by_probability_cyan = linear_probability(distance_metric_cyan, lower_threshold_cyan, upper_threshold_cyan, is_smaller_then=True)
    # Combine the two probabilities
    total_probability = probability_intersection(close_by_probability_blue, close_by_probability_cyan)
    return total_probability


def OrientSameAsCyanBoxFn_trial_34(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is oriented in the same direction as the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    next_object_orientation = get_pose(next_state, object_id, -1)[..., 3:]
    cyan_box_orientation = get_pose(state, cyan_box_id, -1)[..., 3:]
    # Calculate the difference in orientation using the great circle distance metric.
    orientation_difference = great_circle_distance_metric(next_object_orientation, cyan_box_orientation)
    # Define a threshold for acceptable orientation difference.
    threshold = 0.1  # radians, approximately 5.73 degrees
    probability = threshold_probability(orientation_difference, threshold, is_smaller_then=True)
    return probability


def InFrontOfCyanBoxFn_trial_35(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the blue box is placed in front of the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed in front of the cyan box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(blue_box_pose, cyan_box_pose, in_front)
    lower_threshold = 0.0
    # The direction difference should be positive if the blue box is placed in front of the cyan box.
    is_in_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_in_front_probability


def InFrontOfBlueBoxFn_trial_35(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the red box is placed in front of the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    red_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_pose = get_pose(next_state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed in front of the blue box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(red_box_pose, blue_box_pose, in_front)
    lower_threshold = 0.0
    # The direction difference should be positive if the red box is placed in front of the blue box.
    is_in_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_in_front_probability


def LeftOfRedBoxAlignedFn_trial_36(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed to the left of the red box and aligned with it.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the cyan box is placed to the left of the red box
    left_direction = [0.0, -1.0, 0.0]  # Assuming positive y-axis is "right" and negative y-axis is "left"
    direction_difference = position_diff_along_direction(next_cyan_box_pose, red_box_pose, left_direction)
    left_of_red_box_probability = threshold_probability(direction_difference, 0.0, is_smaller_then=False)
    # Evaluate if the cyan box is aligned with the red box
    alignment_metric = position_metric_normal_to_direction(next_cyan_box_pose, red_box_pose, left_direction)
    alignment_probability = threshold_probability(alignment_metric, 0.05, is_smaller_then=True)  # Allow some tolerance
    # Combine probabilities
    total_probability = probability_intersection(left_of_red_box_probability, alignment_probability)
    return total_probability


def PlaceInFrontOfAndAlignedWithBlueBoxFn_trial_37(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in front of and aligned with the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed in front of the blue box
    direction_vector = build_direction_vector(blue_box_pose, next_object_pose)
    front_metric = position_diff_along_direction(next_object_pose, blue_box_pose, direction=[0, 1, 0])
    front_probability = threshold_probability(front_metric, 0.0, is_smaller_then=False)
    # Evaluate if the red box is aligned with the blue box
    alignment_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, direction=[0, 1, 0])
    alignment_probability = threshold_probability(alignment_metric, 0.05, is_smaller_then=True)
    return probability_intersection(front_probability, alignment_probability)


def PlaceLeftOfRedBoxAlignedFn_trial_38(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed left of the red box and if the two boxes are aligned.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_blue_box_pose = get_pose(next_state, blue_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the blue box is placed left of the red box
    left = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(next_blue_box_pose, red_box_pose, left)
    lower_threshold = 0.0
    # The direction difference should be positive if the blue box is placed left of the red box.
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate if the blue box is aligned with the red box in the y-axis
    y_diff_metric = position_metric_normal_to_direction(next_blue_box_pose, red_box_pose, left)
    lower_threshold = 0.0
    upper_threshold = 0.05
    # The y difference should be as small as possible but no larger than 5cm.
    y_diff_probability = linear_probability(y_diff_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    total_probability = probability_intersection(is_left_probability, y_diff_probability)
    return total_probability


CUSTOM_FNS = {
    "PlaceLeftOfRedBoxAlignedFn_trial_38": PlaceLeftOfRedBoxAlignedFn_trial_38,
    "PlaceInFrontOfAndAlignedWithBlueBoxFn_trial_37": PlaceInFrontOfAndAlignedWithBlueBoxFn_trial_37,
    "LeftOfRedBoxAlignedFn_trial_36": LeftOfRedBoxAlignedFn_trial_36,
    "InFrontOfBlueBoxFn_trial_35": InFrontOfBlueBoxFn_trial_35,
    "InFrontOfCyanBoxFn_trial_35": InFrontOfCyanBoxFn_trial_35,
    "OrientSameAsCyanBoxFn_trial_34": OrientSameAsCyanBoxFn_trial_34,
    "PlaceCloseToBlueAndCyanBoxFn_trial_33": PlaceCloseToBlueAndCyanBoxFn_trial_33,
    "PlaceCloseToCyanBoxFn_trial_33": PlaceCloseToCyanBoxFn_trial_33,
    "FarLeftOfTableFn_trial_32": FarLeftOfTableFn_trial_32,
    "CircleAroundScrewdriverFn_trial_31": CircleAroundScrewdriverFn_trial_31,
    "BehindBlueBoxFn_trial_30": BehindBlueBoxFn_trial_30,
    "InFrontOfRedBoxFn_trial_30": InFrontOfRedBoxFn_trial_30,
    "LeftOfRedBoxFn_trial_29": LeftOfRedBoxFn_trial_29,
    "PlaceInTriangleWithRedAndBlueBoxFn_trial_28": PlaceInTriangleWithRedAndBlueBoxFn_trial_28,
    "PlaceInTriangleWithRedAndCyanBoxFn_trial_28": PlaceInTriangleWithRedAndCyanBoxFn_trial_28,
    "PlaceCyanBoxFarFromRedAndBlueBoxFn_trial_27": PlaceCyanBoxFarFromRedAndBlueBoxFn_trial_27,
    "PlaceRedBoxNextToBlueBoxFn_trial_27": PlaceRedBoxNextToBlueBoxFn_trial_27,
    "PlaceInLineWithRedAndBlueFn_trial_26": PlaceInLineWithRedAndBlueFn_trial_26,
    "PlaceLeftOfRedBoxAlignedFn_trial_25": PlaceLeftOfRedBoxAlignedFn_trial_25,
    "PlaceInFrontOfBlueBoxFn_trial_24": PlaceInFrontOfBlueBoxFn_trial_24,
    "LeftAndAlignedWithRedBoxFn_trial_23": LeftAndAlignedWithRedBoxFn_trial_23,
    "PlaceInFrontOfBlueBoxFn_trial_22": PlaceInFrontOfBlueBoxFn_trial_22,
    "PlaceInFrontOfCyanBoxFn_trial_22": PlaceInFrontOfCyanBoxFn_trial_22,
    "OrientWithCyanBoxFn_trial_21": OrientWithCyanBoxFn_trial_21,
    "PlaceCloseToBlueAndCyanBoxFn_trial_20": PlaceCloseToBlueAndCyanBoxFn_trial_20,
    "PlaceCloseToCyanBoxFn_trial_20": PlaceCloseToCyanBoxFn_trial_20,
    "PlaceToLeftFn_trial_19": PlaceToLeftFn_trial_19,
    "CircleAroundScrewdriverFn_trial_18": CircleAroundScrewdriverFn_trial_18,
    "InFrontOfRedBoxBehindCyanBoxFn_trial_17": InFrontOfRedBoxBehindCyanBoxFn_trial_17,
    "LeftOfRedBoxFn_trial_16": LeftOfRedBoxFn_trial_16,
    "TrianglePlacementCyanFn_trial_15": TrianglePlacementCyanFn_trial_15,
    "TrianglePlacementBlueFn_trial_15": TrianglePlacementBlueFn_trial_15,
    "PlaceFarFromBlueAndRedBoxFn_trial_14": PlaceFarFromBlueAndRedBoxFn_trial_14,
    "PlaceNextToBlueBoxFn_trial_14": PlaceNextToBlueBoxFn_trial_14,
    "AlignBoxesInLineFn_trial_13": AlignBoxesInLineFn_trial_13,
    "LeftAndAlignedWithRedBoxFn_trial_12": LeftAndAlignedWithRedBoxFn_trial_12,
    "PlaceInFrontAndAlignWithBlueBoxFn_trial_11": PlaceInFrontAndAlignWithBlueBoxFn_trial_11,
    "LeftAndAlignedWithRedBoxFn_trial_10": LeftAndAlignedWithRedBoxFn_trial_10,
    "PlaceInLineFn_trial_9": PlaceInLineFn_trial_9,
    "OrientRedBoxLikeCyanBoxFn_trial_8": OrientRedBoxLikeCyanBoxFn_trial_8,
    "OrientBlueBoxLikeCyanBoxFn_trial_8": OrientBlueBoxLikeCyanBoxFn_trial_8,
    "PlaceCloseToBlueBoxFn_trial_7": PlaceCloseToBlueBoxFn_trial_7,
    "PlaceCloseToCyanBoxFn_trial_7": PlaceCloseToCyanBoxFn_trial_7,
    "FarLeftOnTableFn_trial_6": FarLeftOnTableFn_trial_6,
    "PlaceInCircleAroundScrewdriver15cmFn_trial_5": PlaceInCircleAroundScrewdriver15cmFn_trial_5,
    "PlaceBlueBehindCyanFn_trial_4": PlaceBlueBehindCyanFn_trial_4,
    "PlaceBlueInFrontOfRedFn_trial_4": PlaceBlueInFrontOfRedFn_trial_4,
    "LeftOfRedBoxFn_trial_3": LeftOfRedBoxFn_trial_3,
    "TrianglePlacementCyanFn_trial_2": TrianglePlacementCyanFn_trial_2,
    "TrianglePlacementBlueFn_trial_2": TrianglePlacementBlueFn_trial_2,
    "FarFromBlueAndRedBoxFn_trial_1": FarFromBlueAndRedBoxFn_trial_1,
    "NextToBlueBoxFn_trial_1": NextToBlueBoxFn_trial_1,
    "AlignBoxesInLineFn_trial_0": AlignBoxesInLineFn_trial_0,
    "HookHandoverOrientationFn": HookHandoverOrientationFn,
    "ScrewdriverPickFn": ScrewdriverPickFn,
    "ScrewdriverPickActionFn": ScrewdriverPickActionFn,
    "HandoverPositionFn": HandoverPositionFn,
    "HandoverOrientationFn": HandoverOrientationFn,
    "HandoverOrientationAndPositionnFn": HandoverOrientationAndPositionnFn,
    "HandoverVerticalOrientationFn": HandoverVerticalOrientationFn,
    "PlaceLeftOfRedBoxFn": PlaceLeftOfRedBoxFn,
    "PlaceInFrontOfBlueBoxFn": PlaceInFrontOfBlueBoxFn,
    "PlaceDistanceApartBlueBoxFn": PlaceDistanceApartBlueBoxFn,
    "PlaceInLineWithRedAndBlueBoxFn": PlaceInLineWithRedAndBlueBoxFn,
    "PlaceNextToBlueBoxFn": PlaceNextToBlueBoxFn,
    "PlaceFarAwayFromRedAndBlueFn": PlaceFarAwayFromRedAndBlueFn,
    "PlaceNextToRedBox20cmFn": PlaceNextToRedBox20cmFn,
    "PlaceNextToRedBoxAndBlueBox20cmFn": PlaceNextToRedBoxAndBlueBox20cmFn,
    "LeftOfRedBoxFn": LeftOfRedBoxFn,
    "BehindBlueBoxFn": BehindBlueBoxFn,
    "InFrontOfBlueBoxFn": InFrontOfBlueBoxFn,
    "PlaceNextToScrewdriver15cmFn": PlaceNextToScrewdriver15cmFn,
    "FarLeftOfTableFn": FarLeftOfTableFn,
    "CloseToCyanBoxFn": CloseToCyanBoxFn,
    "CloseToCyanAndBlueBoxFn": CloseToCyanAndBlueBoxFn,
    "SameOrientationAsCyanBoxFn": SameOrientationAsCyanBoxFn,
    "InFrontOfCyanBoxFn": InFrontOfCyanBoxFn,
}
