"""Defines custom evaluation functions to be used in the trajectory evaluation of the planner.

Author: Jakob Thumm
Date: 2024-04-15
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
    linear_probability_to_normal_probability,
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
    total_probability = probability_intersection(is_left_probability, x_diff_probability)
    return total_probability


def PlaceLeftOfAndNextToRedBoxFn(
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
    # Evaluate if the object has a deviation normal to the given direction.
    y_diff_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, in_front_of)
    lower_threshold = 0.0
    upper_threshold = 0.05
    x_diff_probability = linear_probability(y_diff_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    total_probability = probability_intersection(is_left_probability, x_diff_probability)
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
    probability = linear_probability_to_normal_probability(
        normal_distance_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
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
    close_by_probability = linear_probability_to_normal_probability(
        distance_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
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
    far_away_probability_red = linear_probability_to_normal_probability(
        distance_metric_red, lower_threshold, upper_threshold, is_smaller_then=False
    )
    # Evaluate if the object is placed far away from the blue box
    distance_metric_blue = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    far_away_probability_blue = linear_probability_to_normal_probability(
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
    smaller_than_ideal_probability = linear_probability_to_normal_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability_to_normal_probability(
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
    smaller_than_ideal_probability = linear_probability_to_normal_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability_to_normal_probability(
        distance_metric, ideal_point, upper_threshold, is_smaller_then=True
    )
    probability_red_box = probability_intersection(smaller_than_ideal_probability, bigger_than_ideal_probability)
    distance_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    smaller_than_ideal_probability = linear_probability_to_normal_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability_to_normal_probability(
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
    smaller_than_ideal_probability = linear_probability_to_normal_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability_to_normal_probability(
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
    closeness_probability = linear_probability_to_normal_probability(
        distance_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
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
    closeness_probability_cyan = linear_probability_to_normal_probability(
        distance_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Evaluate if the object is placed close to the blue box
    distance_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    closeness_probability_blue = linear_probability_to_normal_probability(
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
    orientation_probability = linear_probability_to_normal_probability(
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


EVALUATION_FNS = {
    "HookHandoverOrientationFn": HookHandoverOrientationFn,
    "ScrewdriverPickFn": ScrewdriverPickFn,
    "ScrewdriverPickActionFn": ScrewdriverPickActionFn,
    "HandoverPositionFn": HandoverPositionFn,
    "HandoverOrientationFn": HandoverOrientationFn,
    "HandoverOrientationAndPositionnFn": HandoverOrientationAndPositionnFn,
    "HandoverVerticalOrientationFn": HandoverVerticalOrientationFn,
    "PlaceLeftOfRedBoxFn": PlaceLeftOfRedBoxFn,
    "PlaceLeftOfAndNextToRedBoxFn": PlaceLeftOfAndNextToRedBoxFn,
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
