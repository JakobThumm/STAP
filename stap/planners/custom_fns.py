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
    r"""Evaluates if the object is placed in a straight line left of the red box.

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
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate if the object is placed in a straight line left of the red box.
    straight_line_metric = position_metric_normal_to_direction(next_object_pose, red_box_pose, left)
    lower_threshold = 0.0
    upper_threshold = 0.05
    straight_line_probability = linear_probability(
        straight_line_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    total_probability = probability_intersection(is_left_probability, straight_line_probability)
    return total_probability


def PlaceInFrontOfBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed in a straight line in front of the blue box.

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
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate if the object is placed in a straight line in front of the blue box.
    y_diff_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, in_front_of)
    lower_threshold = 0.0
    upper_threshold = 0.05
    x_diff_probability = linear_probability(y_diff_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    total_probability = probability_intersection(is_left_probability, x_diff_probability)
    return total_probability


def PlaceLeftOfAndNextToRedBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed in a straight line left of the red box and if the object is placed 10cm next to the red box.

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
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate if the object is placed in a straight line left of the red box.
    x_diff_metric = position_metric_normal_to_direction(next_object_pose, red_box_pose, left)
    lower_threshold = 0.0
    upper_threshold = 0.05
    x_diff_probability = linear_probability(x_diff_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    total_left_probability = probability_intersection(is_left_probability, x_diff_probability)
    # Evaluate if the object is placed 10cm next to the red box.
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.050
    ideal_point = 0.10
    upper_threshold = 0.15
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
    lower_threshold = torch.pi / 8.0
    upper_threshold = torch.pi / 6.0
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
    r"""Evaluates if the cyan box is placed in a line with the red and blue boxes on the table.

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
    # Get the object IDs from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate the direction vector from red to blue box.
    direction_red_to_blue = build_direction_vector(red_box_pose, blue_box_pose)
    # Evaluate if the cyan box is aligned with the red and blue boxes.
    alignment_metric_red = position_metric_normal_to_direction(cyan_box_pose, red_box_pose, direction_red_to_blue)
    alignment_metric_blue = position_metric_normal_to_direction(cyan_box_pose, blue_box_pose, direction_red_to_blue)
    # Combine the alignment metrics.
    alignment_probability_red = linear_probability(alignment_metric_red, 0.0, 0.05, is_smaller_then=True)
    alignment_probability_blue = linear_probability(alignment_metric_blue, 0.0, 0.05, is_smaller_then=True)
    total_alignment_probability = probability_intersection(alignment_probability_red, alignment_probability_blue)
    return total_alignment_probability


def CloseToBlueBoxFn_trial_1(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the red box is placed close to the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed action [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    red_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed close to the blue box
    distance_metric = position_norm_metric(red_box_pose, blue_box_pose, norm="L2")
    lower_threshold = 0.0
    upper_threshold = 0.05  # 5cm as close
    close_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return close_probability


def FarFromBothBoxesFn_trial_1(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed far away from both the red and the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed action [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the cyan box is placed far away from both the red and the blue box
    distance_to_red = position_norm_metric(cyan_box_pose, red_box_pose, norm="L2")
    distance_to_blue = position_norm_metric(cyan_box_pose, blue_box_pose, norm="L2")
    lower_threshold = 0.2  # 20cm as far
    upper_threshold = 1.0  # Ideally 100cm but considering table size
    far_from_red_probability = linear_probability(
        distance_to_red, lower_threshold, upper_threshold, is_smaller_then=False
    )
    far_from_blue_probability = linear_probability(
        distance_to_blue, lower_threshold, upper_threshold, is_smaller_then=False
    )
    # Combine probabilities to ensure the cyan box is far from both boxes
    far_probability = probability_intersection(far_from_red_probability, far_from_blue_probability)
    return far_probability


def TriangleFormationBlueFn_trial_2(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in a triangle formation with the red and cyan boxes.

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
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the object is placed in a triangle formation
    distance_metric_red = position_norm_metric(blue_box_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_cyan = position_norm_metric(blue_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    triangle_probability_red = linear_probability(distance_metric_red, 0.20, 0.20, is_smaller_then=False)
    triangle_probability_cyan = linear_probability(distance_metric_cyan, 0.20, 0.20, is_smaller_then=False)
    total_probability = probability_intersection(triangle_probability_red, triangle_probability_cyan)
    return total_probability


def TriangleFormationCyanFn_trial_2(
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
    # Get the object ID from the primitive.
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed in a triangle formation
    distance_metric_red = position_norm_metric(cyan_box_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_blue = position_norm_metric(cyan_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    triangle_probability_red = linear_probability(distance_metric_red, 0.20, 0.20, is_smaller_then=False)
    triangle_probability_blue = linear_probability(distance_metric_blue, 0.20, 0.20, is_smaller_then=False)
    total_probability = probability_intersection(triangle_probability_red, triangle_probability_blue)
    return total_probability


def LeftOfRedBoxFn_trial_3(
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
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed left of the red box
    left_of = [0.0, 1.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, left_of)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed left of the red box.
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
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the blue box, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the red box, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the blue box is placed in front of the red box
    front_direction = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(blue_box_pose, red_box_pose, front_direction)
    lower_threshold = 0.0
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
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the blue box, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the cyan box, the current state is more reliable.
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed behind the cyan box
    behind_direction = [-1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(blue_box_pose, cyan_box_pose, behind_direction)
    lower_threshold = 0.0
    behind_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return behind_probability


def CircleAroundScrewdriver15cmFn_trial_5(
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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the screwdriver ID from the environment.
    screwdriver_id = get_object_id_from_name("screwdriver", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the screwdriver, the current state is more reliable.
    screwdriver_pose = get_pose(state, screwdriver_id, -1)
    # Evaluate if the object is placed in a circle of radius 15cm around the screwdriver.
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
    # Assuming the table's origin is at its center and its dimensions are known.
    table_width = 2.0  # Given in the detected objects description.
    table_edge_x = -table_width / 2  # Left edge x-coordinate.

    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)

    # Extract the x-coordinate of the object's pose.
    object_x = next_object_pose[:, 0]

    # Calculate the distance from the left edge of the table.
    distance_from_left_edge = object_x - table_edge_x

    # Define thresholds for "far to the left" as close to the edge but not exactly at the edge to avoid collisions.
    lower_threshold = 0.0  # Exactly at the edge.
    upper_threshold = 0.10  # Slightly away from the edge to be considered "far left" but safe.

    # Calculate the probability using a linear function where closer to the edge is better.
    probability = linear_probability(distance_from_left_edge, lower_threshold, upper_threshold, is_smaller_then=True)

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
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the object is placed close to the cyan box.
    distance_metric = position_norm_metric(next_object_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.05
    upper_threshold = 0.10
    close_by_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return close_by_probability


def PlaceCloseToBlueAndCyanBoxFn_trial_7(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed close to both the blue and cyan box without them touching.

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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the object is placed close to the blue box.
    distance_metric_blue = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold_blue = 0.05
    upper_threshold_blue = 0.10
    close_by_probability_blue = linear_probability(
        distance_metric_blue, lower_threshold_blue, upper_threshold_blue, is_smaller_then=True
    )
    # Evaluate if the object is placed close to the cyan box.
    distance_metric_cyan = position_norm_metric(next_object_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold_cyan = 0.05
    upper_threshold_cyan = 0.10
    close_by_probability_cyan = linear_probability(
        distance_metric_cyan, lower_threshold_cyan, upper_threshold_cyan, is_smaller_then=True
    )
    # Combine the probabilities to ensure the object is close to both boxes.
    total_probability = probability_intersection(close_by_probability_blue, close_by_probability_cyan)
    return total_probability


def OrientSameAsCyanBoxFn_trial_8(
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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_orientation = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_orientation = get_pose(state, cyan_box_id, -1)
    # Evaluate if the object is oriented in the same direction as the cyan box
    orientation_difference = great_circle_distance_metric(next_object_orientation, cyan_box_orientation)
    lower_threshold = 0.0
    upper_threshold = torch.pi / 8
    orientation_similarity_probability = linear_probability(
        orientation_difference, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return orientation_similarity_probability


def InFrontOfBlueBoxFn_trial_9(
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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed in front of the blue box
    front = [1.0, 0.0, 0.0]
    distance_metric = position_diff_along_direction(next_object_pose, blue_box_pose, front)
    lower_threshold = 0.0
    in_front_probability = threshold_probability(distance_metric, lower_threshold, is_smaller_then=False)
    return in_front_probability


def InFrontOfCyanBoxFn_trial_9(
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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the object is placed in front of the cyan box
    front = [1.0, 0.0, 0.0]
    distance_metric = position_diff_along_direction(next_object_pose, cyan_box_pose, front)
    lower_threshold = 0.0
    in_front_probability = threshold_probability(distance_metric, lower_threshold, is_smaller_then=False)
    return in_front_probability


def StraightLineLeftOfRedBoxFn_trial_10(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed in a straight line left of the red box.

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
    # Get the object ID from the primitive.
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the cyan box is placed in a straight line left of the red box
    left_direction = [0.0, -1.0, 0.0]
    position_difference = position_diff_along_direction(next_cyan_box_pose, red_box_pose, left_direction)
    position_normal_difference = position_metric_normal_to_direction(next_cyan_box_pose, red_box_pose, left_direction)
    # The position difference should be positive if the cyan box is placed left of the red box.
    is_left_probability = threshold_probability(position_difference, 0.0, is_smaller_then=False)
    # The position normal difference should be close to zero if in a straight line.
    straight_line_probability = linear_probability(position_normal_difference, 0.0, 0.05, is_smaller_then=True)
    # Combine probabilities to ensure both conditions are met.
    total_probability = probability_intersection(is_left_probability, straight_line_probability)
    return total_probability


def StraightLineInFrontOfBlueBoxFn_trial_11(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the red box is placed in a straight line in front of the blue box.

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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed in a straight line in front of the blue box
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, [1.0, 0.0, 0.0])
    lower_threshold = 0.0
    upper_threshold = 0.05  # Allow a small margin for alignment
    is_in_straight_line_probability = linear_probability(
        direction_difference, lower_threshold, upper_threshold, is_smaller_then=False
    )
    return is_in_straight_line_probability


def StraightLineInFrontOfRedBoxFn_trial_12(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the blue box is placed in a straight line in front of the red box, 10cm apart.

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
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the blue box is placed in a straight line in front of the red box
    direction = [1.0, 0.0, 0.0]  # Front direction
    distance_metric = position_diff_along_direction(next_blue_box_pose, red_box_pose, direction)
    normal_distance_metric = position_metric_normal_to_direction(next_blue_box_pose, red_box_pose, direction)
    distance_probability = linear_probability(distance_metric, 0.10, 0.10, is_smaller_then=False)
    normal_distance_probability = threshold_probability(normal_distance_metric, 0.01, is_smaller_then=True)
    return probability_intersection(distance_probability, normal_distance_probability)


def AlignInLineWithRedAndBlueBoxFn_trial_13(
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
    # Get the object ID from the primitive.
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate the direction vector from the red box to the blue box.
    direction_vector = build_direction_vector(red_box_pose, blue_box_pose)
    # Calculate the positional difference of the cyan box normal to the direction vector.
    normal_distance_metric = position_metric_normal_to_direction(cyan_box_pose, red_box_pose, direction_vector)
    # The cyan box should be close to the line formed by the red and blue boxes, ideally within 5cm.
    probability = linear_probability(normal_distance_metric, 0.0, 0.05, is_smaller_then=True)
    return probability


def PlaceCloseToBlueBoxFn_trial_14(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed close to the blue box.

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
    # Get the object ID from the primitive.
    red_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed close to the blue box.
    distance_metric = position_norm_metric(red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.05
    ideal_point = 0.10
    upper_threshold = 0.15
    probability = linear_probability(distance_metric, lower_threshold, ideal_point, is_smaller_then=False)
    return probability


def PlaceFarFromRedAndBlueBoxFn_trial_14(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed far away from both the red and the blue box.

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
    # Get the object ID from the primitive.
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the cyan box is placed far away from both the red and the blue box.
    distance_metric_red = position_norm_metric(cyan_box_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_blue = position_norm_metric(cyan_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.20
    ideal_point = 1.00
    upper_threshold = 1.20
    probability_red = linear_probability(distance_metric_red, lower_threshold, ideal_point, is_smaller_then=False)
    probability_blue = linear_probability(distance_metric_blue, lower_threshold, ideal_point, is_smaller_then=False)
    # Combine the probabilities to ensure the cyan box is far from both boxes.
    total_probability = probability_intersection(probability_red, probability_blue)
    return total_probability


def TriangleFormationWithRedBoxFn_trial_15(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in a triangle formation with the red box at a distance of 20 cm.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed action [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed at a distance of 20 cm from the red box.
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.20
    upper_threshold = 0.25
    distance_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return distance_probability


def TriangleFormationWithBlueBoxFn_trial_15(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in a triangle formation with the blue and red box at a distance of 20 cm.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed action [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed at a distance of 20 cm from both the blue and red boxes.
    distance_metric_blue = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_red = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.20
    upper_threshold = 0.25
    distance_probability_blue = linear_probability(
        distance_metric_blue, lower_threshold, upper_threshold, is_smaller_then=True
    )
    distance_probability_red = linear_probability(
        distance_metric_red, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine the probabilities to ensure the object is equidistant from both boxes.
    total_probability = probability_intersection(distance_probability_blue, distance_probability_red)
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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed left of the red box
    left_of = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, left_of)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed left of the red box.
    is_left_of_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_left_of_probability


def InFrontOfRedBoxFn_trial_17(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed in front of the red box.

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
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed in front of the red box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, in_front)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed in front of the red box.
    is_in_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_in_front_probability


def BehindBlueBoxFn_trial_17(
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
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed behind the blue box
    behind = [-1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, behind)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed behind the blue box.
    is_behind_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_behind_probability


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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    screwdriver_id = get_object_id_from_name("screwdriver", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    screwdriver_pose = get_pose(state, screwdriver_id, -1)
    # Evaluate if the object is placed in a circle of radius 15 cm around the screwdriver
    distance_metric = position_norm_metric(next_object_pose, screwdriver_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.15 - 0.05  # 15 cm radius minus tolerance
    upper_threshold = 0.15 + 0.05  # 15 cm radius plus tolerance
    circle_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return circle_probability


def PlaceAsFarLeftAsPossibleFn_trial_19(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed as far to the left of the table as possible.

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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the table ID from the environment.
    table_id = get_object_id_from_name("table", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the table, the current state is more reliable.
    table_pose = get_pose(state, table_id, -1)
    # Evaluate how far to the left the object is placed on the table.
    left_direction = [0.0, -1.0, 0.0]
    distance_to_left_edge = position_diff_along_direction(next_object_pose, table_pose, left_direction)
    # Assuming the table width is 2.0m, and the object should be as far to the left as possible.
    lower_threshold = -1.0  # Far left edge of the table
    upper_threshold = 0.0  # Center of the table
    probability = linear_probability(distance_to_left_edge, lower_threshold, upper_threshold, is_smaller_then=False)
    return probability


def PlaceCloseToCyanAndRedBoxFn_trial_20(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the blue box is placed close to both the cyan and red boxes without touching.

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
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the blue box is placed close to the cyan box.
    distance_metric_cyan = position_norm_metric(blue_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    # Evaluate if the blue box is placed close to the red box.
    distance_metric_red = position_norm_metric(blue_box_pose, red_box_pose, norm="L2", axes=["x", "y"])
    # Define thresholds for "close but not touching".
    lower_threshold = 0.05
    upper_threshold = 0.10
    close_to_cyan_probability = linear_probability(
        distance_metric_cyan, lower_threshold, upper_threshold, is_smaller_then=True
    )
    close_to_red_probability = linear_probability(
        distance_metric_red, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine the probabilities to ensure the blue box is close to both the cyan and red boxes.
    total_probability = probability_intersection(close_to_cyan_probability, close_to_red_probability)
    return total_probability


def PlaceCloseToCyanAndBlueBoxFn_trial_20(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the red box is placed close to both the cyan and blue boxes without touching.

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
    # Get the object ID from the primitive.
    red_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed close to the cyan box.
    distance_metric_cyan = position_norm_metric(red_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    # Evaluate if the red box is placed close to the blue box.
    distance_metric_blue = position_norm_metric(red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    # Define thresholds for "close but not touching".
    lower_threshold = 0.05
    upper_threshold = 0.10
    close_to_cyan_probability = linear_probability(
        distance_metric_cyan, lower_threshold, upper_threshold, is_smaller_then=True
    )
    close_to_blue_probability = linear_probability(
        distance_metric_blue, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine the probabilities to ensure the red box is close to both the cyan and blue boxes.
    total_probability = probability_intersection(close_to_cyan_probability, close_to_blue_probability)
    return total_probability


def OrientSameAsCyanBoxFn_trial_21(
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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the object is oriented in the same direction as the cyan box
    orientation_difference = great_circle_distance_metric(next_object_pose, cyan_box_pose)
    # The orientation difference should be as small as possible but no larger than pi/6 for acceptable alignment.
    lower_threshold = 0.0
    upper_threshold = torch.pi / 6
    probability = linear_probability(orientation_difference, lower_threshold, upper_threshold, is_smaller_then=True)
    return probability


def InFrontOfCyanBoxFn_trial_22(
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
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed in front of the cyan box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(blue_box_pose, cyan_box_pose, in_front)
    lower_threshold = 0.0
    # The direction difference should be positive if the blue box is placed in front of the cyan box.
    is_in_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_in_front_probability


def InFrontOfBlueBoxFn_trial_22(
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
    # Get the object ID from the primitive.
    red_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed in front of the blue box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(red_box_pose, blue_box_pose, in_front)
    lower_threshold = 0.0
    # The direction difference should be positive if the red box is placed in front of the blue box.
    is_in_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_in_front_probability


def StraightLineLeftOfRedBoxFn_trial_23(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed in a straight line left of the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed action [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the cyan box is placed in a straight line left of the red box
    left_direction = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(cyan_box_pose, red_box_pose, left_direction)
    # Use linear probability to evaluate the placement
    lower_threshold = 0.0  # At least 0cm to the left
    upper_threshold = 0.2  # Ideally 20cm to the left
    is_straight_line_left_probability = linear_probability(
        direction_difference, lower_threshold, upper_threshold, is_smaller_then=False
    )
    return is_straight_line_left_probability


def PlaceInFrontOfBlueBoxFn_trial_24(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in a straight line in front of the blue box.

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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed in front of the blue box
    front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, front)
    lower_threshold = 0.0
    is_in_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate if the object is placed in a straight line in front of the blue box.
    straight_line_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, front)
    lower_threshold = 0.0
    upper_threshold = 0.05
    straight_line_probability = linear_probability(
        straight_line_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    total_probability = probability_intersection(is_in_front_probability, straight_line_probability)
    return total_probability


def PlaceInFrontOfRedBox10cmFn_trial_25(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in a straight line in front of the red box, 10cm apart.

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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed in front of the red box
    front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, front)
    lower_threshold = 0.05
    upper_threshold = 0.15
    in_front_probability = linear_probability(
        direction_difference, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Evaluate if the object is placed in a straight line in front of the red box.
    straight_line_metric = position_metric_normal_to_direction(next_object_pose, red_box_pose, front)
    straight_line_probability = linear_probability(straight_line_metric, 0.0, 0.05, is_smaller_then=True)
    total_probability = probability_intersection(in_front_probability, straight_line_probability)
    return total_probability


def PlaceInLineWithRedAndBlueFn_trial_26(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in a line with the red and blue boxes on the table.

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
    # Get the object ID from the primitive.
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate the direction vector from the red box to the blue box.
    direction_vector = build_direction_vector(red_box_pose, blue_box_pose)
    # Evaluate if the cyan box is placed in a line with the red and blue boxes.
    line_metric_red = position_metric_normal_to_direction(cyan_box_pose, red_box_pose, direction_vector)
    line_metric_blue = position_metric_normal_to_direction(cyan_box_pose, blue_box_pose, direction_vector)
    # Combine the metrics to evaluate the placement in line.
    lower_threshold = 0.0
    upper_threshold = 0.05
    in_line_probability_red = linear_probability(
        line_metric_red, lower_threshold, upper_threshold, is_smaller_then=True
    )
    in_line_probability_blue = linear_probability(
        line_metric_blue, lower_threshold, upper_threshold, is_smaller_then=True
    )
    total_probability = probability_intersection(in_line_probability_red, in_line_probability_blue)
    return total_probability


def PlaceFarFromRedAndBlueBoxFn_trial_27(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed far away from both the red and the blue box.

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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed far from the red box.
    distance_metric_red = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_blue = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.20  # Starting point for "far".
    upper_threshold = 1.00  # Ideally 100cm, but considering table size.
    far_probability_red = linear_probability(
        distance_metric_red, lower_threshold, upper_threshold, is_smaller_then=False
    )
    far_probability_blue = linear_probability(
        distance_metric_blue, lower_threshold, upper_threshold, is_smaller_then=False
    )
    # Combine probabilities to ensure the object is far from both boxes.
    return probability_intersection(far_probability_red, far_probability_blue)


def TriangleFormationWithRedBoxFn_trial_28(
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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Calculate the distance between the placed object and the red box.
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2")
    # The distance should be close to 20cm.
    lower_threshold = 0.20 - 0.05  # Allowing some tolerance
    upper_threshold = 0.20 + 0.05
    probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return probability


def TriangleFormationWithBlueBoxFn_trial_28(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in a triangle formation with the blue box and red box.

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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Calculate the distance between the placed object and the blue box.
    distance_to_blue = position_norm_metric(next_object_pose, blue_box_pose, norm="L2")
    # Calculate the distance between the placed object and the red box.
    distance_to_red = position_norm_metric(next_object_pose, red_box_pose, norm="L2")
    # The distance should be close to 20cm.
    lower_threshold = 0.20 - 0.05  # Allowing some tolerance
    upper_threshold = 0.20 + 0.05
    probability_blue = linear_probability(distance_to_blue, lower_threshold, upper_threshold, is_smaller_then=True)
    probability_red = linear_probability(distance_to_red, lower_threshold, upper_threshold, is_smaller_then=True)
    # Combine the probabilities to ensure the object is in a triangle formation with both boxes.
    total_probability = probability_intersection(probability_blue, probability_red)
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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed left of the red box
    left = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, left)
    lower_threshold = 0.0
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_left_probability


def InFrontOfRedBoxFn_trial_30(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the blue box is placed in front of the red box.

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
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the blue box is placed in front of the red box
    front_direction = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(blue_box_pose, red_box_pose, front_direction)
    mean = 0.0
    std = 0.1
    in_front_probability = normal_probability(direction_difference, mean, std, is_smaller_then=True)
    return in_front_probability


def BehindCyanBoxFn_trial_30(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the blue box is placed behind the cyan box.

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
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed behind the cyan box
    behind_direction = [-1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(blue_box_pose, cyan_box_pose, behind_direction)
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
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the screwdriver object ID from the environment.
    screwdriver_id = get_object_id_from_name("screwdriver", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the screwdriver, the current state is more reliable.
    screwdriver_pose = get_pose(state, screwdriver_id, -1)
    # Evaluate if the object is placed in a circle of radius 15 cm around the screwdriver
    distance_metric = position_norm_metric(next_object_pose, screwdriver_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.15 - 0.05  # 15 cm radius minus tolerance
    upper_threshold = 0.15 + 0.05  # 15 cm radius plus tolerance
    circle_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # Assuming the table's origin is at its center and its width is 2.0 meters,
    # the far left of the table would be at -1.0 meters on the y-axis.
    table_left_edge_pose = generate_pose_batch([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], next_object_pose)
    # Evaluate how far the object is to the left of the table
    distance_metric = position_norm_metric(next_object_pose, table_left_edge_pose, norm="L2", axes=["y"])
    lower_threshold = 0.0  # As close to the edge as possible
    upper_threshold = 0.5  # Reasonably far from the edge but still on the left side
    far_left_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return far_left_probability


def PlaceCloseToCyanAndRedBoxFn_trial_33(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed close to both the cyan and red box without touching.

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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed close to the cyan box.
    distance_metric_cyan = position_norm_metric(next_object_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10
    upper_threshold = 0.20
    close_to_cyan_probability = linear_probability(
        distance_metric_cyan, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Evaluate if the object is placed close to the red box.
    distance_metric_red = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    close_to_red_probability = linear_probability(
        distance_metric_red, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine the two probabilities
    total_probability = probability_intersection(close_to_cyan_probability, close_to_red_probability)
    return total_probability


def PlaceCloseToBlueAndCyanBoxFn_trial_33(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed close to both the blue and cyan box without touching.

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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the object is placed close to the blue box.
    distance_metric_blue = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10
    upper_threshold = 0.20
    close_to_blue_probability = linear_probability(
        distance_metric_blue, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Evaluate if the object is placed close to the cyan box.
    distance_metric_cyan = position_norm_metric(next_object_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    close_to_cyan_probability = linear_probability(
        distance_metric_cyan, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine the two probabilities
    total_probability = probability_intersection(close_to_blue_probability, close_to_cyan_probability)
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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the cyan box ID from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the cyan box, the current state is more reliable.
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the object is oriented in the same direction as the cyan box.
    orientation_difference = great_circle_distance_metric(next_object_pose, cyan_box_pose)
    # Define thresholds for orientation similarity.
    lower_threshold = 0.0  # No difference is ideal.
    upper_threshold = torch.pi / 6  # Up to 30 degrees difference is acceptable.
    probability = linear_probability(orientation_difference, lower_threshold, upper_threshold, is_smaller_then=True)
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
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed in front of the cyan box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(blue_box_pose, cyan_box_pose, in_front)
    lower_threshold = 0.0
    upper_threshold = 0.05
    is_in_front_probability = linear_probability(
        direction_difference, lower_threshold, upper_threshold, is_smaller_then=False
    )
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
    # Get the object ID from the primitive.
    red_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed in front of the blue box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(red_box_pose, blue_box_pose, in_front)
    lower_threshold = 0.0
    upper_threshold = 0.05
    is_in_front_probability = linear_probability(
        direction_difference, lower_threshold, upper_threshold, is_smaller_then=False
    )
    return is_in_front_probability


def LeftOfRedBoxFn_trial_36(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed to the left of the red box.

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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed to the left of the red box
    left_of = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, left_of)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed to the left of the red box.
    is_left_of_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_left_of_probability


def PlaceInFrontOfBlueBoxFn_trial_37(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in a straight line in front of the blue box.

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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed in a straight line in front of the blue box.
    direction = [1, 0, 0]  # Front direction
    normal_distance_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, direction)
    # The distance normal to the direction should be minimal.
    lower_threshold = 0.0
    upper_threshold = 0.05
    probability_normal_to_direction = linear_probability(
        normal_distance_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Evaluate the distance along the direction to ensure it's placed in front but not too far.
    distance_along_direction = position_diff_along_direction(next_object_pose, blue_box_pose, direction)
    ideal_distance = 0.20  # Ideal distance in front of the blue box
    lower_threshold_distance = 0.15
    upper_threshold_distance = 0.25
    probability_along_direction = linear_probability(
        distance_along_direction, lower_threshold_distance, upper_threshold_distance, is_smaller_then=False
    )
    return probability_intersection(probability_normal_to_direction, probability_along_direction)


def StraightLineInFrontOfRedBoxFn_trial_38(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in a straight line in front of the red box, 10cm apart.

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
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_blue_box_pose = get_pose(next_state, blue_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the blue box is placed in front of the red box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_blue_box_pose, red_box_pose, in_front)
    # Evaluate if the blue box is 10cm apart from the red box
    distance_metric = position_norm_metric(next_blue_box_pose, red_box_pose, norm="L2", axes=["x", "y", "z"])
    distance_probability = linear_probability(distance_metric, 0.09, 0.11, is_smaller_then=True)
    # Combine probabilities
    total_probability = probability_intersection(direction_difference, distance_probability)
    return total_probability


CUSTOM_FNS = {
    "StraightLineInFrontOfRedBoxFn_trial_38": StraightLineInFrontOfRedBoxFn_trial_38,
    "PlaceInFrontOfBlueBoxFn_trial_37": PlaceInFrontOfBlueBoxFn_trial_37,
    "LeftOfRedBoxFn_trial_36": LeftOfRedBoxFn_trial_36,
    "InFrontOfBlueBoxFn_trial_35": InFrontOfBlueBoxFn_trial_35,
    "InFrontOfCyanBoxFn_trial_35": InFrontOfCyanBoxFn_trial_35,
    "OrientSameAsCyanBoxFn_trial_34": OrientSameAsCyanBoxFn_trial_34,
    "PlaceCloseToBlueAndCyanBoxFn_trial_33": PlaceCloseToBlueAndCyanBoxFn_trial_33,
    "PlaceCloseToCyanAndRedBoxFn_trial_33": PlaceCloseToCyanAndRedBoxFn_trial_33,
    "FarLeftOfTableFn_trial_32": FarLeftOfTableFn_trial_32,
    "CircleAroundScrewdriverFn_trial_31": CircleAroundScrewdriverFn_trial_31,
    "BehindCyanBoxFn_trial_30": BehindCyanBoxFn_trial_30,
    "InFrontOfRedBoxFn_trial_30": InFrontOfRedBoxFn_trial_30,
    "LeftOfRedBoxFn_trial_29": LeftOfRedBoxFn_trial_29,
    "TriangleFormationWithBlueBoxFn_trial_28": TriangleFormationWithBlueBoxFn_trial_28,
    "TriangleFormationWithRedBoxFn_trial_28": TriangleFormationWithRedBoxFn_trial_28,
    "PlaceFarFromRedAndBlueBoxFn_trial_27": PlaceFarFromRedAndBlueBoxFn_trial_27,
    "PlaceInLineWithRedAndBlueFn_trial_26": PlaceInLineWithRedAndBlueFn_trial_26,
    "PlaceInFrontOfRedBox10cmFn_trial_25": PlaceInFrontOfRedBox10cmFn_trial_25,
    "PlaceInFrontOfBlueBoxFn_trial_24": PlaceInFrontOfBlueBoxFn_trial_24,
    "StraightLineLeftOfRedBoxFn_trial_23": StraightLineLeftOfRedBoxFn_trial_23,
    "InFrontOfBlueBoxFn_trial_22": InFrontOfBlueBoxFn_trial_22,
    "InFrontOfCyanBoxFn_trial_22": InFrontOfCyanBoxFn_trial_22,
    "OrientSameAsCyanBoxFn_trial_21": OrientSameAsCyanBoxFn_trial_21,
    "PlaceCloseToCyanAndBlueBoxFn_trial_20": PlaceCloseToCyanAndBlueBoxFn_trial_20,
    "PlaceCloseToCyanAndRedBoxFn_trial_20": PlaceCloseToCyanAndRedBoxFn_trial_20,
    "PlaceAsFarLeftAsPossibleFn_trial_19": PlaceAsFarLeftAsPossibleFn_trial_19,
    "CircleAroundScrewdriverFn_trial_18": CircleAroundScrewdriverFn_trial_18,
    "BehindBlueBoxFn_trial_17": BehindBlueBoxFn_trial_17,
    "InFrontOfRedBoxFn_trial_17": InFrontOfRedBoxFn_trial_17,
    "LeftOfRedBoxFn_trial_16": LeftOfRedBoxFn_trial_16,
    "TriangleFormationWithBlueBoxFn_trial_15": TriangleFormationWithBlueBoxFn_trial_15,
    "TriangleFormationWithRedBoxFn_trial_15": TriangleFormationWithRedBoxFn_trial_15,
    "PlaceFarFromRedAndBlueBoxFn_trial_14": PlaceFarFromRedAndBlueBoxFn_trial_14,
    "PlaceCloseToBlueBoxFn_trial_14": PlaceCloseToBlueBoxFn_trial_14,
    "AlignInLineWithRedAndBlueBoxFn_trial_13": AlignInLineWithRedAndBlueBoxFn_trial_13,
    "StraightLineInFrontOfRedBoxFn_trial_12": StraightLineInFrontOfRedBoxFn_trial_12,
    "StraightLineInFrontOfBlueBoxFn_trial_11": StraightLineInFrontOfBlueBoxFn_trial_11,
    "StraightLineLeftOfRedBoxFn_trial_10": StraightLineLeftOfRedBoxFn_trial_10,
    "InFrontOfCyanBoxFn_trial_9": InFrontOfCyanBoxFn_trial_9,
    "InFrontOfBlueBoxFn_trial_9": InFrontOfBlueBoxFn_trial_9,
    "OrientSameAsCyanBoxFn_trial_8": OrientSameAsCyanBoxFn_trial_8,
    "PlaceCloseToBlueAndCyanBoxFn_trial_7": PlaceCloseToBlueAndCyanBoxFn_trial_7,
    "PlaceCloseToCyanBoxFn_trial_7": PlaceCloseToCyanBoxFn_trial_7,
    "FarLeftOnTableFn_trial_6": FarLeftOnTableFn_trial_6,
    "CircleAroundScrewdriver15cmFn_trial_5": CircleAroundScrewdriver15cmFn_trial_5,
    "PlaceBlueBehindCyanFn_trial_4": PlaceBlueBehindCyanFn_trial_4,
    "PlaceBlueInFrontOfRedFn_trial_4": PlaceBlueInFrontOfRedFn_trial_4,
    "LeftOfRedBoxFn_trial_3": LeftOfRedBoxFn_trial_3,
    "TriangleFormationCyanFn_trial_2": TriangleFormationCyanFn_trial_2,
    "TriangleFormationBlueFn_trial_2": TriangleFormationBlueFn_trial_2,
    "FarFromBothBoxesFn_trial_1": FarFromBothBoxesFn_trial_1,
    "CloseToBlueBoxFn_trial_1": CloseToBlueBoxFn_trial_1,
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
    "PlaceLeftOfAndNextToRedBoxFn": PlaceLeftOfAndNextToRedBoxFn,
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
