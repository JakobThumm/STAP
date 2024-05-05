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
    r"""Evaluates if the cyan box is placed in line with the red and blue boxes on the table.

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
    # Calculate the direction vector from the red box to the blue box
    direction_vector_red_to_blue = build_direction_vector(red_box_pose, blue_box_pose)
    # Calculate the positional difference of the cyan box along the direction vector
    position_diff_cyan_along_direction = position_diff_along_direction(
        cyan_box_pose, red_box_pose, direction_vector_red_to_blue
    )
    # Calculate the positional difference of the cyan box normal to the direction vector
    position_diff_cyan_normal_to_direction = position_metric_normal_to_direction(
        cyan_box_pose, red_box_pose, direction_vector_red_to_blue
    )
    # Evaluate if the cyan box is aligned with the red and blue boxes
    alignment_probability = linear_probability(position_diff_cyan_normal_to_direction, 0.0, 0.05, is_smaller_then=True)
    # Evaluate if the cyan box is placed in line between the red and blue boxes
    in_line_probability = linear_probability(position_diff_cyan_along_direction, 0.10, 0.20, is_smaller_then=True)
    # Combine the probabilities
    total_probability = probability_intersection(alignment_probability, in_line_probability)
    return total_probability


def CloseToBlueBoxFn_trial_1(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the preference probability of placing the red box close to the blue box.
    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Predicted next state after executing this action.
        primitive: Optional primitive to receive the object information from
    Returns:
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate the distance between the red box and the blue box
    distance_metric = position_norm_metric(red_box_pose, blue_box_pose, norm="L2")
    # Define thresholds based on the rules
    lower_threshold = 0.05  # 5cm as close
    upper_threshold = 0.10  # 10cm as the upper limit for being considered close
    # Calculate the probability
    probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return probability


def FarFromRedAndBlueBoxFn_trial_1(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the preference probability of placing the cyan box far from both the red and the blue box.
    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Predicted next state after executing this action.
        primitive: Optional primitive to receive the object information from
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
    # Calculate the distance from the cyan box to both the red and blue boxes
    distance_to_red = position_norm_metric(cyan_box_pose, red_box_pose, norm="L2")
    distance_to_blue = position_norm_metric(cyan_box_pose, blue_box_pose, norm="L2")
    # Define thresholds based on the rules
    lower_threshold = 0.20  # 20cm as the lower limit for being considered far
    upper_threshold = 1.00  # 100cm as the ideal distance
    # Calculate the probabilities
    probability_red = linear_probability(distance_to_red, lower_threshold, upper_threshold, is_smaller_then=False)
    probability_blue = linear_probability(distance_to_blue, lower_threshold, upper_threshold, is_smaller_then=False)
    # Combine the probabilities to ensure the cyan box is far from both the red and blue boxes
    total_probability = probability_intersection(probability_red, probability_blue)
    return total_probability


def TrianglePlacementBlueFn_trial_2(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed to form a triangle with the red and cyan boxes.

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
    # Evaluate if the object is placed to form a triangle
    distance_metric_red = position_norm_metric(blue_box_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_cyan = position_norm_metric(blue_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.20
    upper_threshold = 0.25
    triangle_probability_red = linear_probability(
        distance_metric_red, lower_threshold, upper_threshold, is_smaller_then=True
    )
    triangle_probability_cyan = linear_probability(
        distance_metric_cyan, lower_threshold, upper_threshold, is_smaller_then=True
    )
    total_probability = probability_intersection(triangle_probability_red, triangle_probability_cyan)
    return total_probability


def TrianglePlacementCyanFn_trial_2(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed to form a triangle with the red and blue boxes.

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
    # Evaluate if the object is placed to form a triangle
    distance_metric_red = position_norm_metric(cyan_box_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_blue = position_norm_metric(cyan_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.20
    upper_threshold = 0.25
    triangle_probability_red = linear_probability(
        distance_metric_red, lower_threshold, upper_threshold, is_smaller_then=True
    )
    triangle_probability_blue = linear_probability(
        distance_metric_blue, lower_threshold, upper_threshold, is_smaller_then=True
    )
    total_probability = probability_intersection(triangle_probability_red, triangle_probability_blue)
    return total_probability


def LeftOfRedBoxFn_trial_3(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the preference probability of placing an object left of the red box.
    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Predicted next state after executing this action.
        primitive: Optional primitive to receive the object information from
    Returns:
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
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
    # Evaluate if the object is placed left of the red box.
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, direction=[0, 1, 0])
    lower_threshold = 0.10  # At least 10cm to the left to be considered 'left'.
    upper_threshold = 0.20  # Ideally 20cm to the left.
    probability = linear_probability(direction_difference, lower_threshold, upper_threshold, is_smaller_then=True)
    return probability


def PlaceCyanBoxInFrontOfRedBoxFn_trial_4(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in front of the red box but behind the blue box.

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
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the cyan box is placed in front of the red box
    front_direction = [1.0, 0.0, 0.0]  # Assuming front is in the positive x-direction
    distance_in_front_of_red = position_diff_along_direction(cyan_box_pose, red_box_pose, front_direction)
    in_front_of_red_probability = threshold_probability(distance_in_front_of_red, 0.0, is_smaller_then=False)
    # Evaluate if the cyan box is placed behind the blue box
    distance_behind_blue = position_diff_along_direction(blue_box_pose, cyan_box_pose, front_direction)
    behind_blue_probability = threshold_probability(distance_behind_blue, 0.0, is_smaller_then=False)
    # Combine the two probabilities
    total_probability = probability_intersection(in_front_of_red_probability, behind_blue_probability)
    return total_probability


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
    ideal_radius = 0.15  # 15cm
    lower_threshold = 0.10  # 10cm
    upper_threshold = 0.20  # 20cm
    probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return probability


def FarLeftOnTableFn_trial_6(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed as far to the left of the table as possible.

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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # Assuming the leftmost position on the table is at x = 0.
    leftmost_position = 0.0
    # The x position of the object in the next state.
    object_x_position = next_object_pose[:, 0]
    # The farther to the left, the better. Use a linear probability with a preference for x positions close to 0.
    lower_threshold = 0.0  # Ideal position.
    upper_threshold = 0.5  # Acceptable but not ideal.
    probability = linear_probability(object_x_position, lower_threshold, upper_threshold, is_smaller_then=True)
    return probability


def PlaceCloseToCyanAndRedBoxFn_trial_7(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed close to both the cyan and red boxes without them touching.

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
    close_to_cyan_probability = linear_probability(distance_metric_cyan, 0.05, 0.10, is_smaller_then=True)
    # Evaluate if the blue box is placed close to the red box.
    distance_metric_red = position_norm_metric(blue_box_pose, red_box_pose, norm="L2", axes=["x", "y"])
    close_to_red_probability = linear_probability(distance_metric_red, 0.05, 0.10, is_smaller_then=True)
    # Combine the probabilities
    total_probability = probability_intersection(close_to_cyan_probability, close_to_red_probability)
    return total_probability


def PlaceCloseToCyanAndBlueBoxFn_trial_7(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed close to both the cyan and blue boxes without them touching.

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
    close_to_cyan_probability = linear_probability(distance_metric_cyan, 0.05, 0.10, is_smaller_then=True)
    # Evaluate if the red box is placed close to the blue box.
    distance_metric_blue = position_norm_metric(red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    close_to_blue_probability = linear_probability(distance_metric_blue, 0.05, 0.10, is_smaller_then=True)
    # Combine the probabilities
    total_probability = probability_intersection(close_to_cyan_probability, close_to_blue_probability)
    return total_probability


def OrientLikeCyanBoxFn_trial_8(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is oriented in the same direction as the cyan box.

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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the cyan box ID from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_orientation = get_pose(next_state, object_id, -1)[..., 3:]
    # For the cyan box, the current state is more reliable.
    cyan_box_orientation = get_pose(state, cyan_box_id, -1)[..., 3:]
    # Calculate the orientation difference using the great circle distance metric.
    orientation_difference = great_circle_distance_metric(next_object_orientation, cyan_box_orientation)
    # Define thresholds based on the rules.
    small_difference = torch.pi / 3
    ideal_difference = torch.pi / 6
    # Calculate the probability that the orientation satisfies the human's preferences.
    orientation_probability = linear_probability(
        orientation_difference, ideal_difference, small_difference, is_smaller_then=True
    )
    return orientation_probability


def PlaceBlueBoxInLineFn_trial_9(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in line with the cyan box, maintaining a distance of at least 10cm.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed in line and at least 10cm from the cyan box
    distance_metric = position_norm_metric(blue_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10
    upper_threshold = 0.20
    distance_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return distance_probability


def PlaceRedBoxInLineFn_trial_9(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in line with the blue box, maintaining a distance of at least 10cm.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    red_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed in line and at least 10cm from the blue box
    distance_metric = position_norm_metric(red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10
    upper_threshold = 0.20
    distance_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return distance_probability


def LeftOfRedBoxAndAlignedFn_trial_10(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed left of the red box and aligned with it.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Predicted next state after executing this action.
        primitive: Optional primitive to receive the object information from
    Returns:
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object IDs from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # Get the poses from the state.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the cyan box is placed left of the red box.
    left_direction = [0.0, -1.0, 0.0]
    position_difference_left = position_diff_along_direction(cyan_box_pose, red_box_pose, left_direction)
    is_left_probability = threshold_probability(position_difference_left, 0.0, is_smaller_then=False)
    # Evaluate if the cyan box is aligned with the red box.
    alignment_direction = [1.0, 0.0, 0.0]  # Considering alignment along the x-axis.
    position_difference_alignment = position_metric_normal_to_direction(
        cyan_box_pose, red_box_pose, alignment_direction
    )
    alignment_probability = linear_probability(position_difference_alignment, 0.05, 0.10, is_smaller_then=True)
    # Combine probabilities.
    total_probability = probability_intersection(is_left_probability, alignment_probability)
    return total_probability


def PlaceInFrontAndAlignedWithBlueBoxFn_trial_11(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in front of the blue box and aligned with it.

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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed in front of the blue box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, in_front)
    front_probability = threshold_probability(direction_difference, 0.0, is_smaller_then=False)
    # Evaluate if the object is aligned with the blue box
    normal_distance_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, in_front)
    alignment_probability = linear_probability(normal_distance_metric, 0.0, 0.05, is_smaller_then=True)
    # Combine the probabilities
    total_probability = probability_intersection(front_probability, alignment_probability)
    return total_probability


def InFrontAndAlignedWithRedBoxFn_trial_12(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in front of and aligned with the red box, 10cm apart.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Predicted next state after executing this action.
        primitive: Optional primitive to receive the object information from
    Returns:
        The probability that action `a` on primitive satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the blue box is placed in front of the red box, 10cm apart.
    front_direction = [1.0, 0.0, 0.0]  # Assuming the front is along the positive x-axis
    distance_metric = position_diff_along_direction(blue_box_pose, red_box_pose, front_direction)
    distance_probability = linear_probability(distance_metric, 0.10, 0.10, is_smaller_then=False)
    # Evaluate if the blue box is aligned with the red box along the y-axis.
    alignment_metric = position_metric_normal_to_direction(blue_box_pose, red_box_pose, front_direction)
    alignment_probability = linear_probability(alignment_metric, 0.0, 0.05, is_smaller_then=True)
    # Combine the probabilities for being in front and aligned.
    total_probability = probability_intersection(distance_probability, alignment_probability)
    return total_probability


def AlignBoxesInLineFn_trial_13(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the preference probability of placing the cyan box in line with the red and blue boxes.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Predicted next state after executing this action.
        primitive: Optional primitive to receive the object information from
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

    # Get the poses of the objects.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)

    # Calculate the direction vector from the red box to the blue box.
    direction_vector = build_direction_vector(red_box_pose, blue_box_pose)

    # Calculate the positional difference of the cyan box normal to the direction vector.
    position_diff_metric = position_metric_normal_to_direction(cyan_box_pose, red_box_pose, direction_vector)

    # Calculate the probability that the cyan box is aligned with the red and blue boxes.
    # Considering a tolerance of 5cm for being considered "in line".
    probability_in_line = linear_probability(position_diff_metric, 0, 0.05, is_smaller_then=True)

    return probability_in_line


def PlaceCloseToBlueBoxFn_trial_14(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed close to the blue box, within a distance of 5cm to 10cm.

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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed close to the blue box.
    distance_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.05  # 5cm
    upper_threshold = 0.10  # 10cm
    return linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)


def PlaceFarFromRedAndBlueBoxFn_trial_14(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed far away from both the red and the blue box, ideally more than 100cm apart.

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
    # Evaluate if the object is placed far away from both the red and the blue box.
    distance_to_red_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_to_blue_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.20  # 20cm
    upper_threshold = 1.00  # 100cm
    probability_red = linear_probability(distance_to_red_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    probability_blue = linear_probability(
        distance_to_blue_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine the probabilities to ensure the object is far from both boxes.
    return probability_intersection(probability_red, probability_blue)


def TriangleFormationWithRedBoxFn_trial_15(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in a triangle formation with the red box at a 20cm edge length.

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
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed at a 20cm distance from the red box.
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    ideal_distance = 0.20
    distance_probability = linear_probability(
        distance_metric, ideal_distance - 0.05, ideal_distance + 0.05, is_smaller_then=False
    )
    return distance_probability


def TriangleFormationWithBlueBoxFn_trial_15(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in a triangle formation with the blue and red box at a 20cm edge length.

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
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed at a 20cm distance from both the blue and red boxes.
    distance_metric_blue = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_red = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    ideal_distance = 0.20
    distance_probability_blue = linear_probability(
        distance_metric_blue, ideal_distance - 0.05, ideal_distance + 0.05, is_smaller_then=False
    )
    distance_probability_red = linear_probability(
        distance_metric_red, ideal_distance - 0.05, ideal_distance + 0.05, is_smaller_then=False
    )
    # Combine the probabilities to ensure the cyan box is equidistant from both the blue and red boxes.
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
    left_of = [-1.0, 0.0, 0.0]
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
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed in front of the red box
    in_front_of_red = position_diff_along_direction(blue_box_pose, red_box_pose, [1.0, 0.0, 0.0])
    in_front_of_red_probability = linear_probability(in_front_of_red, 0.05, 0.10, is_smaller_then=False)
    # Evaluate if the blue box is placed behind the cyan box
    behind_cyan = position_diff_along_direction(cyan_box_pose, blue_box_pose, [1.0, 0.0, 0.0])
    behind_cyan_probability = linear_probability(behind_cyan, 0.05, 0.10, is_smaller_then=False)
    # Combine the probabilities
    total_probability = probability_intersection(in_front_of_red_probability, behind_cyan_probability)
    return total_probability


def CircleAroundScrewdriverFn_trial_18(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the preference probability of placing an object in a circle of radius 15 cm around the screwdriver.
    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Predicted next state after executing this action.
        primitive: Optional primitive to receive the object information from
    Returns:
        The probability that action `a` on primitive {{Primitive.name}} satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
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
    # Evaluate if the object is placed in a circle of radius 15 cm around the screwdriver
    distance_metric = position_norm_metric(next_object_pose, screwdriver_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.15  # 15 cm
    upper_threshold = 0.20  # 20 cm for some flexibility
    circle_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return circle_probability


def PlaceBlueBoxToLeftFn_trial_19(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the preference for placing the blue box as far to the left of the table as possible.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Predicted next state after executing this action.
        primitive: Optional primitive to receive the object information from

    Returns:
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
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
    # Calculate the distance from the left edge of the table.
    left_edge_distance = position_diff_along_direction(next_object_pose, table_pose, [-1.0, 0.0, 0.0])
    # Use linear probability to encourage placing the object as far to the left as possible.
    probability = linear_probability(left_edge_distance, 0.05, 0.20, is_smaller_then=True)
    return probability


def PlaceRedBoxToLeftFn_trial_19(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the preference for placing the red box as far to the left of the table as possible.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Predicted next state after executing this action.
        primitive: Optional primitive to receive the object information from

    Returns:
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
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
    # Calculate the distance from the left edge of the table.
    left_edge_distance = position_diff_along_direction(next_object_pose, table_pose, [-1.0, 0.0, 0.0])
    # Use linear probability to encourage placing the object as far to the left as possible.
    probability = linear_probability(left_edge_distance, 0.05, 0.20, is_smaller_then=True)
    return probability


def PlaceCloseToCyanBoxFn_trial_20(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed close to the cyan box without them touching.

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
    lower_threshold = 0.10  # 10cm as close but not touching
    upper_threshold = 0.15  # 15cm as the upper limit for being considered close
    close_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return close_probability


def PlaceCloseToBlueBoxFn_trial_20(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed close to the blue box without them touching.

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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed close to the blue box.
    distance_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10  # 10cm as close but not touching
    upper_threshold = 0.15  # 15cm as the upper limit for being considered close
    close_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return close_probability


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
    next_object_orientation = get_pose(next_state, object_id, -1)[3:]
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_orientation = get_pose(state, cyan_box_id, -1)[3:]
    # Calculate the difference in orientation using the great circle distance metric.
    orientation_difference = great_circle_distance_metric(next_object_orientation, cyan_box_orientation)
    # We consider differences of torch.pi/3 as small, but ideally torch.pi/6.
    lower_threshold = torch.pi / 6
    upper_threshold = torch.pi / 3
    probability = linear_probability(orientation_difference, lower_threshold, upper_threshold, is_smaller_then=True)
    return probability


def InFrontOfRedBoxFn_trial_22(
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
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the blue box is placed in front of the red box
    distance_metric = position_norm_metric(blue_box_pose, red_box_pose, norm="L2", axes=["x"])
    lower_threshold = 0.10  # Close distance
    upper_threshold = 0.20  # Ideal distance
    in_front_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return in_front_probability


def InFrontOfBlueBoxFn_trial_22(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in front of the blue box.

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
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the cyan box is placed in front of the blue box
    distance_metric = position_norm_metric(cyan_box_pose, blue_box_pose, norm="L2", axes=["x"])
    lower_threshold = 0.10  # Close distance
    upper_threshold = 0.20  # Ideal distance
    in_front_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return in_front_probability


def LeftOfRedBoxAlignedFn_trial_23(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the preference probability of placing the cyan box left of the red box and aligned with it.
    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Predicted next state after executing this action.
        primitive: Optional primitive to receive the object information from
    Returns:
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object IDs from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # Get the poses from the state.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Calculate the direction difference to evaluate if the cyan box is left of the red box.
    left_direction = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(cyan_box_pose, red_box_pose, left_direction)
    is_left_probability = linear_probability(direction_difference, 0.0, 0.1, is_smaller_then=False)
    # Calculate the alignment along the y-axis.
    alignment_difference = position_metric_normal_to_direction(cyan_box_pose, red_box_pose, left_direction)
    alignment_probability = linear_probability(alignment_difference, 0.0, 0.05)
    # Combine the probabilities.
    total_probability = probability_intersection(is_left_probability, alignment_probability)
    return total_probability


def PlaceInFrontOfBlueBoxFn_trial_24(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in front of the blue box and aligned with it.

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
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed in front of the blue box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, in_front)
    lower_threshold = 0.0
    upper_threshold = 0.10
    in_front_probability = linear_probability(
        direction_difference, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Evaluate if the object is aligned with the blue box in the y direction.
    y_diff_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, in_front)
    y_diff_probability = linear_probability(y_diff_metric, 0.0, 0.05, is_smaller_then=True)
    total_probability = probability_intersection(in_front_probability, y_diff_probability)
    return total_probability


def PlaceInFrontOfRedBoxAligned10cmFn_trial_25(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in front of the red box, aligned, and 10cm apart.

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
    ideal_distance = 0.10  # 10cm apart
    distance_probability = linear_probability(
        direction_difference, ideal_distance - 0.05, ideal_distance + 0.05, is_smaller_then=True
    )
    # Evaluate if the object is aligned with the red box
    alignment_metric = position_metric_normal_to_direction(next_object_pose, red_box_pose, front)
    alignment_probability = linear_probability(alignment_metric, 0.0, 0.05, is_smaller_then=True)
    total_probability = probability_intersection(distance_probability, alignment_probability)
    return total_probability


def AlignBoxesInLineFn_trial_26(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in line with the red and blue boxes on the table.

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
    direction_vector_red_to_blue = build_direction_vector(red_box_pose, blue_box_pose)
    # Calculate the positional difference of the cyan box along the direction vector.
    position_diff_cyan_along_direction = position_diff_along_direction(
        cyan_box_pose, red_box_pose, direction_vector_red_to_blue
    )
    # Calculate the positional difference of the cyan box normal to the direction vector.
    position_diff_cyan_normal_to_direction = position_metric_normal_to_direction(
        cyan_box_pose, red_box_pose, direction_vector_red_to_blue
    )
    # The cyan box should be aligned with the red and blue boxes, so the normal difference should be minimal.
    alignment_probability = linear_probability(position_diff_cyan_normal_to_direction, 0.0, 0.05, is_smaller_then=True)
    # The cyan box should be placed between the red and blue boxes or in line with them, so the along direction difference should be positive.
    in_line_probability = threshold_probability(position_diff_cyan_along_direction, 0.0, is_smaller_then=False)
    # Combine the probabilities to ensure both conditions are met.
    total_probability = probability_intersection(alignment_probability, in_line_probability)
    return total_probability


def CloseToBlueBoxFn_trial_27(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed close to the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Predicted next state after executing this action.
        primitive: Optional primitive to receive the object information from

    Returns:
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    red_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate the distance between the red and blue boxes.
    distance_metric = position_norm_metric(red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    # Define the thresholds for "close" distance.
    lower_threshold = 0.05  # 5cm
    upper_threshold = 0.10  # 10cm
    probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return probability


def FarFromRedAndBlueBoxFn_trial_27(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed far away from both the red and blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Predicted next state after executing this action.
        primitive: Optional primitive to receive the object information from

    Returns:
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate the distance between the cyan box and both the red and blue boxes.
    distance_to_red = position_norm_metric(cyan_box_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_to_blue = position_norm_metric(cyan_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    # Define the thresholds for "far" distance.
    lower_threshold = 0.20  # 20cm
    upper_threshold = 1.00  # 100cm
    probability_red = linear_probability(distance_to_red, lower_threshold, upper_threshold, is_smaller_then=False)
    probability_blue = linear_probability(distance_to_blue, lower_threshold, upper_threshold, is_smaller_then=False)
    # Combine the probabilities to ensure the cyan box is far from both the red and blue boxes.
    return probability_intersection(probability_red, probability_blue)


def FormTriangleWithRedBoxFn_trial_28(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed to form a triangle with the red and cyan boxes with 20cm edges.

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
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Calculate distances
    distance_to_red = position_norm_metric(next_object_pose, red_box_pose, "L2")
    distance_to_cyan = position_norm_metric(next_object_pose, cyan_box_pose, "L2")
    # Evaluate distances to form a triangle
    lower_threshold = 0.2  # 20cm
    upper_threshold = 0.3  # 30cm for some flexibility
    probability_to_red = linear_probability(distance_to_red, lower_threshold, upper_threshold, is_smaller_then=True)
    probability_to_cyan = linear_probability(distance_to_cyan, lower_threshold, upper_threshold, is_smaller_then=True)
    total_probability = probability_intersection(probability_to_red, probability_to_cyan)
    return total_probability


def FormTriangleWithBlueBoxFn_trial_28(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed to form a triangle with the red and blue boxes with 20cm edges.

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
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate distances
    distance_to_red = position_norm_metric(next_object_pose, red_box_pose, "L2")
    distance_to_blue = position_norm_metric(next_object_pose, blue_box_pose, "L2")
    # Evaluate distances to form a triangle
    lower_threshold = 0.2  # 20cm
    upper_threshold = 0.3  # 30cm for some flexibility
    probability_to_red = linear_probability(distance_to_red, lower_threshold, upper_threshold, is_smaller_then=True)
    probability_to_blue = linear_probability(distance_to_blue, lower_threshold, upper_threshold, is_smaller_then=True)
    total_probability = probability_intersection(probability_to_red, probability_to_blue)
    return total_probability


def LeftOfRedBoxFn_trial_29(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed left of the red box.

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
    # The direction difference should be positive if the object is placed left of the red box.
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_left_probability


def InFrontOfRedBoxFn_trial_30(
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
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the blue box is placed in front of the red box.
    direction_difference = position_diff_along_direction(blue_box_pose, red_box_pose, [1, 0, 0])
    lower_threshold = 0.10  # 10cm
    upper_threshold = 0.20  # 20cm
    in_front_probability = linear_probability(
        direction_difference, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return in_front_probability


def BehindCyanBoxFn_trial_30(
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
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed behind the cyan box.
    direction_difference = position_diff_along_direction(cyan_box_pose, blue_box_pose, [1, 0, 0])
    lower_threshold = 0.10  # 10cm
    upper_threshold = 0.20  # 20cm
    behind_probability = linear_probability(
        direction_difference, lower_threshold, upper_threshold, is_smaller_then=True
    )
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
    radius = 0.15  # 15 cm
    lower_threshold = radius - 0.05  # Slightly smaller than the radius to allow for some flexibility
    upper_threshold = radius + 0.05  # Slightly larger than the radius to allow for some flexibility
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
    # Assuming the table's left edge is at x = 0 (normalized), evaluate how close the object is to this edge.
    # The x-coordinate of the object's pose represents its distance from the left edge.
    # Use a linear probability function to evaluate how close the object is to the left edge.
    # The closer to the edge, the higher the probability.
    left_edge_distance = next_object_pose[..., 0]  # Extract the x-coordinate
    # Assuming the table width is 3.0, we define the far left as within 0.1 of the table's left edge.
    lower_threshold = 0.0  # Closest possible to the left edge
    upper_threshold = 0.1  # Slightly further away but still considered far left
    far_left_probability = linear_probability(
        left_edge_distance, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return far_left_probability


def PlaceCloseToBlueBoxFn_trial_33(
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
    blue_box_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed close to the cyan box.
    distance_metric = position_norm_metric(blue_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10  # Close but not touching
    upper_threshold = 0.15  # Ideal close distance
    close_by_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return close_by_probability


def PlaceCloseToRedBoxFn_trial_33(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed close to the cyan and blue boxes without them touching.

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
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed close to the cyan box.
    distance_metric_cyan = position_norm_metric(red_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_blue = position_norm_metric(red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10  # Close but not touching
    upper_threshold = 0.15  # Ideal close distance
    close_by_probability_cyan = linear_probability(
        distance_metric_cyan, lower_threshold, upper_threshold, is_smaller_then=True
    )
    close_by_probability_blue = linear_probability(
        distance_metric_blue, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine the probabilities to ensure the red box is close to both the cyan and blue boxes.
    total_probability = probability_intersection(close_by_probability_cyan, close_by_probability_blue)
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
    # Get the non-manipulated object IDs from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the object is oriented in the same direction as the cyan box.
    orientation_difference = great_circle_distance_metric(next_object_pose, cyan_box_pose)
    # Use linear probability to evaluate the orientation difference.
    lower_threshold = torch.pi / 6  # Ideally the same orientation.
    upper_threshold = torch.pi / 3  # Acceptable orientation difference.
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
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed in front of the cyan box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(blue_box_pose, cyan_box_pose, in_front)
    # Using linear probability to evaluate the distance
    is_in_front_probability = linear_probability(direction_difference, 0.10, 0.20, is_smaller_then=False)
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
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    red_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed in front of the blue box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(red_box_pose, blue_box_pose, in_front)
    # Using linear probability to evaluate the distance
    is_in_front_probability = linear_probability(direction_difference, 0.10, 0.20, is_smaller_then=False)
    return is_in_front_probability


def LeftOfRedBoxAlignedFn_trial_36(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the preference probability of placing the cyan box left of the red box and aligned with it.
    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Predicted next state after executing this action.
        primitive: Optional primitive to receive the object information from
    Returns:
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object IDs from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # Get the poses from the state.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Calculate the positional difference along the x-axis (left-right direction).
    left_of_red_box_metric = position_diff_along_direction(cyan_box_pose, red_box_pose, direction=[-1, 0, 0])
    # Calculate the positional difference along the y-axis (front-back direction) for alignment.
    alignment_metric = position_metric_normal_to_direction(cyan_box_pose, red_box_pose, direction=[0, 1, 0])
    # Define thresholds for being left and aligned.
    left_threshold = 0.0  # Positive value means left of the red box.
    alignment_threshold = 0.05  # Within 5cm to be considered aligned.
    # Calculate probabilities.
    left_probability = threshold_probability(left_of_red_box_metric, left_threshold, is_smaller_then=False)
    alignment_probability = linear_probability(alignment_metric, 0, alignment_threshold, is_smaller_then=True)
    # Combine probabilities.
    total_probability = probability_intersection(left_probability, alignment_probability)
    return total_probability


def AlignAndPlaceInFrontOfBlueBoxFn_trial_37(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in front of the blue box and aligned with it.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Predicted next state after executing this action.
        primitive: Optional primitive to receive the object information from

    Returns:
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    red_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed in front of the blue box.
    direction_vector = build_direction_vector(blue_box_pose, red_box_pose)
    front_metric = position_diff_along_direction(blue_box_pose, red_box_pose, direction_vector)
    front_probability = linear_probability(front_metric, 0.05, 0.10, is_smaller_then=False)
    # Evaluate if the red box is aligned with the blue box.
    normal_distance_metric = position_metric_normal_to_direction(red_box_pose, blue_box_pose, direction_vector)
    alignment_probability = linear_probability(normal_distance_metric, 0.0, 0.05, is_smaller_then=True)
    # Combine probabilities for being in front and aligned.
    total_probability = probability_intersection(front_probability, alignment_probability)
    return total_probability


def AlignAndFrontOfRedBoxFn_trial_38(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in front of the red box, aligned, and 10cm apart.

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
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the blue box is placed in front of the red box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(blue_box_pose, red_box_pose, in_front)
    # The direction difference should be around 10cm.
    distance_probability = linear_probability(direction_difference, 0.1, 0.2, is_smaller_then=True)
    # Evaluate if the blue box is aligned with the red box
    y_diff_metric = position_metric_normal_to_direction(blue_box_pose, red_box_pose, in_front)
    alignment_probability = linear_probability(y_diff_metric, 0.0, 0.05, is_smaller_then=True)
    total_probability = probability_intersection(distance_probability, alignment_probability)
    return total_probability


CUSTOM_FNS = {
    "AlignAndFrontOfRedBoxFn_trial_38": AlignAndFrontOfRedBoxFn_trial_38,
    "AlignAndPlaceInFrontOfBlueBoxFn_trial_37": AlignAndPlaceInFrontOfBlueBoxFn_trial_37,
    "LeftOfRedBoxAlignedFn_trial_36": LeftOfRedBoxAlignedFn_trial_36,
    "InFrontOfBlueBoxFn_trial_35": InFrontOfBlueBoxFn_trial_35,
    "InFrontOfCyanBoxFn_trial_35": InFrontOfCyanBoxFn_trial_35,
    "OrientSameAsCyanBoxFn_trial_34": OrientSameAsCyanBoxFn_trial_34,
    "PlaceCloseToRedBoxFn_trial_33": PlaceCloseToRedBoxFn_trial_33,
    "PlaceCloseToBlueBoxFn_trial_33": PlaceCloseToBlueBoxFn_trial_33,
    "FarLeftOfTableFn_trial_32": FarLeftOfTableFn_trial_32,
    "CircleAroundScrewdriverFn_trial_31": CircleAroundScrewdriverFn_trial_31,
    "BehindCyanBoxFn_trial_30": BehindCyanBoxFn_trial_30,
    "InFrontOfRedBoxFn_trial_30": InFrontOfRedBoxFn_trial_30,
    "LeftOfRedBoxFn_trial_29": LeftOfRedBoxFn_trial_29,
    "FormTriangleWithBlueBoxFn_trial_28": FormTriangleWithBlueBoxFn_trial_28,
    "FormTriangleWithRedBoxFn_trial_28": FormTriangleWithRedBoxFn_trial_28,
    "FarFromRedAndBlueBoxFn_trial_27": FarFromRedAndBlueBoxFn_trial_27,
    "CloseToBlueBoxFn_trial_27": CloseToBlueBoxFn_trial_27,
    "AlignBoxesInLineFn_trial_26": AlignBoxesInLineFn_trial_26,
    "PlaceInFrontOfRedBoxAligned10cmFn_trial_25": PlaceInFrontOfRedBoxAligned10cmFn_trial_25,
    "PlaceInFrontOfBlueBoxFn_trial_24": PlaceInFrontOfBlueBoxFn_trial_24,
    "LeftOfRedBoxAlignedFn_trial_23": LeftOfRedBoxAlignedFn_trial_23,
    "InFrontOfBlueBoxFn_trial_22": InFrontOfBlueBoxFn_trial_22,
    "InFrontOfRedBoxFn_trial_22": InFrontOfRedBoxFn_trial_22,
    "OrientSameAsCyanBoxFn_trial_21": OrientSameAsCyanBoxFn_trial_21,
    "PlaceCloseToBlueBoxFn_trial_20": PlaceCloseToBlueBoxFn_trial_20,
    "PlaceCloseToCyanBoxFn_trial_20": PlaceCloseToCyanBoxFn_trial_20,
    "PlaceRedBoxToLeftFn_trial_19": PlaceRedBoxToLeftFn_trial_19,
    "PlaceBlueBoxToLeftFn_trial_19": PlaceBlueBoxToLeftFn_trial_19,
    "CircleAroundScrewdriverFn_trial_18": CircleAroundScrewdriverFn_trial_18,
    "InFrontOfRedBoxBehindCyanBoxFn_trial_17": InFrontOfRedBoxBehindCyanBoxFn_trial_17,
    "LeftOfRedBoxFn_trial_16": LeftOfRedBoxFn_trial_16,
    "TriangleFormationWithBlueBoxFn_trial_15": TriangleFormationWithBlueBoxFn_trial_15,
    "TriangleFormationWithRedBoxFn_trial_15": TriangleFormationWithRedBoxFn_trial_15,
    "PlaceFarFromRedAndBlueBoxFn_trial_14": PlaceFarFromRedAndBlueBoxFn_trial_14,
    "PlaceCloseToBlueBoxFn_trial_14": PlaceCloseToBlueBoxFn_trial_14,
    "AlignBoxesInLineFn_trial_13": AlignBoxesInLineFn_trial_13,
    "InFrontAndAlignedWithRedBoxFn_trial_12": InFrontAndAlignedWithRedBoxFn_trial_12,
    "PlaceInFrontAndAlignedWithBlueBoxFn_trial_11": PlaceInFrontAndAlignedWithBlueBoxFn_trial_11,
    "LeftOfRedBoxAndAlignedFn_trial_10": LeftOfRedBoxAndAlignedFn_trial_10,
    "PlaceRedBoxInLineFn_trial_9": PlaceRedBoxInLineFn_trial_9,
    "PlaceBlueBoxInLineFn_trial_9": PlaceBlueBoxInLineFn_trial_9,
    "OrientLikeCyanBoxFn_trial_8": OrientLikeCyanBoxFn_trial_8,
    "PlaceCloseToCyanAndBlueBoxFn_trial_7": PlaceCloseToCyanAndBlueBoxFn_trial_7,
    "PlaceCloseToCyanAndRedBoxFn_trial_7": PlaceCloseToCyanAndRedBoxFn_trial_7,
    "FarLeftOnTableFn_trial_6": FarLeftOnTableFn_trial_6,
    "PlaceInCircleAroundScrewdriver15cmFn_trial_5": PlaceInCircleAroundScrewdriver15cmFn_trial_5,
    "PlaceCyanBoxInFrontOfRedBoxFn_trial_4": PlaceCyanBoxInFrontOfRedBoxFn_trial_4,
    "LeftOfRedBoxFn_trial_3": LeftOfRedBoxFn_trial_3,
    "TrianglePlacementCyanFn_trial_2": TrianglePlacementCyanFn_trial_2,
    "TrianglePlacementBlueFn_trial_2": TrianglePlacementBlueFn_trial_2,
    "FarFromRedAndBlueBoxFn_trial_1": FarFromRedAndBlueBoxFn_trial_1,
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
