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
    total_probability = probability_intersection(is_left_probability, x_diff_probability)
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
    # Calculate the direction vector from red to blue box
    direction_red_to_blue = build_direction_vector(red_box_pose, blue_box_pose)
    # Calculate the positional difference of the cyan box along the direction from red to blue
    position_diff_cyan_along_direction = position_diff_along_direction(
        cyan_box_pose, red_box_pose, direction_red_to_blue
    )
    # Calculate the positional difference of the cyan box normal to the direction from red to blue
    position_diff_cyan_normal_to_direction = position_metric_normal_to_direction(
        cyan_box_pose, red_box_pose, direction_red_to_blue
    )
    # The cyan box should be aligned with the red and blue boxes, thus the normal difference should be minimal
    normal_diff_probability = linear_probability(
        position_diff_cyan_normal_to_direction, 0.0, 0.05, is_smaller_then=True
    )
    # The cyan box should be between the red and blue boxes, thus the along direction difference should be positive but not too large
    along_diff_lower_threshold = 0.10  # At least 10cm from the red box
    along_diff_upper_threshold = 0.20  # Up to 20cm from the red box, considering the boxes are in a line
    along_diff_probability = linear_probability(
        position_diff_cyan_along_direction, along_diff_lower_threshold, along_diff_upper_threshold, is_smaller_then=True
    )
    # Combine the probabilities
    total_probability = probability_intersection(normal_diff_probability, along_diff_probability)
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
    # Get the poses of the objects.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate the distance between the red box and the blue box.
    distance_metric = position_norm_metric(red_box_pose, blue_box_pose, norm="L2")
    # Define thresholds for "close" distance.
    lower_threshold = 0.05  # 5cm
    upper_threshold = 0.10  # 10cm
    # Calculate the probability based on the distance metric.
    probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return probability


def FarFromBothBoxesFn_trial_1(
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
    # Get the poses of the objects.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate the distance between the cyan box and the other two boxes.
    distance_to_red = position_norm_metric(cyan_box_pose, red_box_pose, norm="L2")
    distance_to_blue = position_norm_metric(cyan_box_pose, blue_box_pose, norm="L2")
    # Define thresholds for "far" distance.
    lower_threshold = 0.20  # 20cm
    upper_threshold = 1.00  # 100cm
    # Calculate the probability based on the distance metrics.
    probability_red = linear_probability(distance_to_red, lower_threshold, upper_threshold, is_smaller_then=False)
    probability_blue = linear_probability(distance_to_blue, lower_threshold, upper_threshold, is_smaller_then=False)
    # Combine the probabilities to ensure the cyan box is far from both boxes.
    total_probability = probability_intersection(probability_red, probability_blue)
    return total_probability


def TriangleFormationWithRedBoxFn_trial_2(
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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed at a distance of 20cm from the red box.
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    ideal_distance = 0.20
    distance_probability = linear_probability(
        distance_metric, ideal_distance - 0.05, ideal_distance + 0.05, is_smaller_then=False
    )
    return distance_probability


def TriangleFormationWithBlueBoxFn_trial_2(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in a triangle formation with the blue and red box at a distance of 20cm.

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
    # Evaluate if the object is placed at a distance of 20cm from both the blue and red boxes.
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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed to the left of the red box.
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, [0, 1, 0])
    lower_threshold = 0.10  # At least 10cm to the left to be considered "left of"
    is_left_of_probability = linear_probability(direction_difference, lower_threshold, 0.20, is_smaller_then=False)
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
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the blue box is placed in front of the red box
    front_direction = [1.0, 0.0, 0.0]  # Assuming front is along the positive x-axis
    direction_difference = position_diff_along_direction(blue_box_pose, red_box_pose, front_direction)
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
        Evaluation of the performed placement [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed behind the cyan box
    behind_direction = [-1.0, 0.0, 0.0]  # Assuming behind is along the negative x-axis
    direction_difference = position_diff_along_direction(blue_box_pose, cyan_box_pose, behind_direction)
    lower_threshold = 0.0
    # The direction difference should be positive if the blue box is placed behind the cyan box.
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
    lower_threshold = 0.10  # 10cm as close
    ideal_point = 0.15  # 15cm as the ideal radius
    upper_threshold = 0.20  # 20cm as far
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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # Assuming the left side of the table is at x = 0
    left_side_of_table = 0.0
    # Calculate the distance from the left side of the table
    distance_from_left = next_object_pose[..., 0] - left_side_of_table
    # We consider far to the left starting at 20cm, but ideally 100cm.
    lower_threshold = 0.20  # 20cm
    upper_threshold = 1.00  # 100cm
    probability = linear_probability(distance_from_left, lower_threshold, upper_threshold, is_smaller_then=False)
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
    lower_threshold = 0.10  # Minimum distance to be considered close but not touching.
    upper_threshold = 0.20  # Maximum distance to still be considered close.
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
    lower_threshold = 0.10  # Minimum distance to be considered close but not touching.
    upper_threshold = 0.20  # Maximum distance to still be considered close.
    close_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return close_probability


def OrientLikeCyanBoxFn_trial_8(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the blue box is oriented in the same direction as the cyan box.

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
    # Evaluate if the blue box is oriented in the same direction as the cyan box
    orientation_metric = great_circle_distance_metric(blue_box_pose, cyan_box_pose)
    # Use a linear probability function to evaluate the orientation similarity
    lower_threshold = torch.pi / 6.0
    upper_threshold = torch.pi / 3.0
    orientation_probability = linear_probability(
        orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return orientation_probability


def OrientLikeCyanAndBlueBoxFn_trial_8(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the red box is oriented in the same direction as the cyan and blue box.

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
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is oriented in the same direction as the cyan and blue box
    orientation_metric_cyan = great_circle_distance_metric(red_box_pose, cyan_box_pose)
    orientation_metric_blue = great_circle_distance_metric(red_box_pose, blue_box_pose)
    # Use a linear probability function to evaluate the orientation similarity
    lower_threshold = torch.pi / 6.0
    upper_threshold = torch.pi / 3.0
    orientation_probability_cyan = linear_probability(
        orientation_metric_cyan, lower_threshold, upper_threshold, is_smaller_then=True
    )
    orientation_probability_blue = linear_probability(
        orientation_metric_blue, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine the probabilities to ensure the red box is oriented like both the cyan and blue boxes
    total_orientation_probability = probability_intersection(orientation_probability_cyan, orientation_probability_blue)
    return total_orientation_probability


def LinePlacementBlueFrontOfRedFn_trial_9(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in front of the red box.

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
    front = [1.0, 0.0, 0.0]
    distance_metric = position_diff_along_direction(blue_box_pose, red_box_pose, front)
    lower_threshold = 0.10
    upper_threshold = 0.20
    front_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return front_probability


def LinePlacementRedFrontOfCyanFn_trial_9(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in front of the cyan box.

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
    # Get the non-manipulated object IDs from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the red box is placed in front of the cyan box
    front = [1.0, 0.0, 0.0]
    distance_metric = position_diff_along_direction(red_box_pose, cyan_box_pose, front)
    lower_threshold = 0.10
    upper_threshold = 0.20
    front_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return front_probability


def LeftOfAndAlignedWithRedBoxFn_trial_10(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed left of and aligned with the red box.

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
    # Evaluate if the cyan box is placed left of the red box
    left_direction = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(next_cyan_box_pose, red_box_pose, left_direction)
    is_left_probability = threshold_probability(direction_difference, 0.0, is_smaller_then=False)
    # Evaluate if the cyan box is aligned with the red box
    alignment_metric = position_metric_normal_to_direction(next_cyan_box_pose, red_box_pose, left_direction)
    alignment_probability = linear_probability(alignment_metric, 0.05, 0.10, is_smaller_then=True)
    # Combine probabilities
    total_probability = probability_intersection(is_left_probability, alignment_probability)
    return total_probability


def PlaceInFrontAndAlignedWithBlueBoxFn_trial_11(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in front of and aligned with the blue box.

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
    in_front_probability = threshold_probability(direction_difference, 0.0, is_smaller_then=False)
    # Evaluate if the object is aligned with the blue box
    normal_distance_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, in_front)
    aligned_probability = linear_probability(normal_distance_metric, 0.0, 0.05, is_smaller_then=True)
    # Combine the probabilities
    total_probability = probability_intersection(in_front_probability, aligned_probability)
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
    # Get the object ID from the primitive.
    blue_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the blue box is placed left of the red box
    left_direction = [0.0, -1.0, 0.0]
    left_metric = position_diff_along_direction(next_blue_box_pose, red_box_pose, left_direction)
    left_probability = linear_probability(left_metric, 0.10, 0.20, is_smaller_then=False)
    # Evaluate if the blue box is aligned with the red box
    alignment_metric = position_metric_normal_to_direction(next_blue_box_pose, red_box_pose, left_direction)
    alignment_probability = linear_probability(alignment_metric, 0.0, 0.05, is_smaller_then=True)
    # Combine probabilities
    total_probability = probability_intersection(left_probability, alignment_probability)
    return total_probability


def AlignInLineWithBoxesFn_trial_13(
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
    # The cyan box should be close to the line formed by the red and blue boxes, within 5cm.
    probability_normal_distance = linear_probability(normal_distance_metric, 0.0, 0.05, is_smaller_then=True)
    # Calculate the distance along the direction vector to ensure the cyan box is between the red and blue boxes.
    along_distance_red_to_cyan = position_diff_along_direction(red_box_pose, cyan_box_pose, direction_vector)
    along_distance_cyan_to_blue = position_diff_along_direction(cyan_box_pose, blue_box_pose, direction_vector)
    # Both distances should be positive, indicating the cyan box is between the red and blue boxes.
    probability_along_distance_red_to_cyan = threshold_probability(
        along_distance_red_to_cyan, 0.0, is_smaller_then=False
    )
    probability_along_distance_cyan_to_blue = threshold_probability(
        along_distance_cyan_to_blue, 0.0, is_smaller_then=False
    )
    # Combine the probabilities to ensure the cyan box is in line and between the red and blue boxes.
    return probability_intersection(
        probability_normal_distance,
        probability_intersection(probability_along_distance_red_to_cyan, probability_along_distance_cyan_to_blue),
    )


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
    r"""Evaluates if the cyan box is placed far from both the red and the blue box, ideally more than 100cm away.

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
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed far from both the red and the blue box.
    distance_metric_red = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_blue = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.20  # 20cm
    upper_threshold = 1.00  # 100cm
    probability_red = linear_probability(distance_metric_red, lower_threshold, upper_threshold, is_smaller_then=True)
    probability_blue = linear_probability(distance_metric_blue, lower_threshold, upper_threshold, is_smaller_then=True)
    # Combine the probabilities to ensure the object is far from both boxes.
    return probability_intersection(probability_red, probability_blue)


def TriangleFormationWithRedBoxFn_trial_15(
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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed at a distance of 20cm from the red box.
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    ideal_distance = 0.20
    distance_probability = linear_probability(
        distance_metric, ideal_distance - 0.05, ideal_distance + 0.05, is_smaller_then=False
    )
    return distance_probability


def TriangleFormationWithBlueBoxFn_trial_15(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in a triangle formation with the blue box at a distance of 20cm.

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
    # Evaluate if the object is placed at a distance of 20cm from the blue box.
    distance_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    ideal_distance = 0.20
    distance_probability = linear_probability(
        distance_metric, ideal_distance - 0.05, ideal_distance + 0.05, is_smaller_then=False
    )
    return distance_probability


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
    left_direction = [-1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, left_direction)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed left of the red box.
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_left_probability


def PlaceInFrontOfRedBoxFn_trial_17(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed in front of the red box.

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
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed in front of the red box
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, in_front)
    lower_threshold = 0.10  # At least 10cm in front
    upper_threshold = 0.20  # Ideally 20cm in front
    probability = linear_probability(direction_difference, lower_threshold, upper_threshold, is_smaller_then=False)
    return probability


def PlaceBehindCyanBoxFn_trial_17(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed behind the cyan box.

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
    # Evaluate if the object is placed behind the cyan box
    behind = [-1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, cyan_box_pose, behind)
    lower_threshold = 0.10  # At least 10cm behind
    upper_threshold = 0.20  # Ideally 20cm behind
    probability = linear_probability(direction_difference, lower_threshold, upper_threshold, is_smaller_then=False)
    return probability


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
    lower_threshold = 0.15  # 15 cm
    upper_threshold = 0.20  # 20 cm for some tolerance
    circle_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return circle_probability


def PlaceToLeftOfTableFn_trial_19(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the preference for placing an object as far to the left of the table as possible.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: Optional primitive to receive the object information from

    Returns:
        The probability that action `a` on primitive satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Assuming the table's leftmost position is at x = 0, and its width is 3.000 meters.
    table_width = 3.000
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # Extract the x position of the object.
    object_x_position = next_object_pose[:, 0]
    # Define the ideal and farthest left position as 0.05 meters from the left edge of the table.
    ideal_left_position = 0.05
    # Define the threshold for being considered far left as 0.5 meters from the left edge.
    far_left_threshold = 0.5
    # Calculate the linear probability based on the object's x position.
    probability = linear_probability(
        metric=object_x_position,
        lower_threshold=ideal_left_position,
        upper_threshold=far_left_threshold,
        is_smaller_then=True,
    )
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
    blue_box_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed close to the cyan box.
    distance_metric = position_norm_metric(blue_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10  # 10cm as close but not touching
    upper_threshold = 0.15  # 15cm to allow some flexibility
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
    red_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed close to the blue box.
    distance_metric = position_norm_metric(red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10  # 10cm as close but not touching
    upper_threshold = 0.15  # 15cm to allow some flexibility
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
    next_object_orientation = get_pose(next_state, object_id, -1)[..., 3:]
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_orientation = get_pose(state, cyan_box_id, -1)[..., 3:]
    # Calculate the difference in orientation using the great circle distance metric.
    orientation_difference = great_circle_distance_metric(next_object_orientation, cyan_box_orientation)
    # We consider a small orientation difference as close to being aligned.
    lower_threshold = torch.pi / 6.0
    upper_threshold = torch.pi / 3.0
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
    lower_threshold = 0.10  # 10cm as close
    upper_threshold = 0.20  # 20cm as far
    in_front_probability = linear_probability(
        direction_difference, lower_threshold, upper_threshold, is_smaller_then=False
    )
    return in_front_probability


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
    lower_threshold = 0.10  # 10cm as close
    upper_threshold = 0.20  # 20cm as far
    in_front_probability = linear_probability(
        direction_difference, lower_threshold, upper_threshold, is_smaller_then=False
    )
    return in_front_probability


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
    # Get the object IDs from the environment.
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # Get the poses from the state.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Calculate the left of the red box metric.
    left_metric = position_diff_along_direction(cyan_box_pose, red_box_pose, [0.0, -1.0, 0.0])
    # Calculate the alignment metric (z-axis difference should be minimal).
    alignment_metric = position_norm_metric(cyan_box_pose, red_box_pose, norm="L2", axes=["z"])
    # Convert metrics to probabilities.
    left_probability = linear_probability(left_metric, 0.0, 0.1, is_smaller_then=False)
    alignment_probability = linear_probability(alignment_metric, 0.0, 0.05)
    # Combine probabilities.
    total_probability = probability_intersection(left_probability, alignment_probability)
    return total_probability


def PlaceInFrontOfBlueBoxFn_trial_24(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in front of the blue box and if the two boxes are aligned.

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
    in_front = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, in_front)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed in front of the blue box.
    is_in_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate if the object has a deviation in the y direction.
    y_diff_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, in_front)
    lower_threshold = 0.0
    upper_threshold = 0.05
    # The y difference should be as small as possible but no larger than 5cm.
    y_diff_probability = linear_probability(y_diff_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    total_in_front_probability = probability_intersection(is_in_front_probability, y_diff_probability)
    return total_in_front_probability


def PlaceLeftOfRedBoxAlignedFn_trial_25(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed left of the red box and ensures they are aligned.

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
    return total_left_probability


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
    # Calculate the direction vector from red to blue box.
    direction_red_to_blue = build_direction_vector(red_box_pose, blue_box_pose)
    # Calculate the positional difference of the cyan box normal to the direction from red to blue.
    normal_diff_cyan = position_metric_normal_to_direction(cyan_box_pose, red_box_pose, direction_red_to_blue)
    # The cyan box should be aligned with the red and blue boxes, so the normal difference should be minimal.
    alignment_probability = linear_probability(normal_diff_cyan, 0.0, 0.05, is_smaller_then=True)
    return alignment_probability


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
    object_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate the distance between the red box and the blue box.
    distance_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    # Define the thresholds for close distance.
    lower_threshold = 0.05  # 5cm
    upper_threshold = 0.10  # 10cm
    probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return probability


def FarFromRedAndBlueBoxFn_trial_27(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed far away from both the red and the blue box.

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
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate the distance between the cyan box and the red and blue boxes.
    distance_metric_red = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_blue = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    # Define the thresholds for far distance.
    lower_threshold = 0.20  # 20cm
    upper_threshold = 1.00  # 100cm
    probability_red = linear_probability(distance_metric_red, lower_threshold, upper_threshold, is_smaller_then=False)
    probability_blue = linear_probability(distance_metric_blue, lower_threshold, upper_threshold, is_smaller_then=False)
    # Combine the probabilities to ensure the cyan box is far from both the red and blue boxes.
    return probability_intersection(probability_red, probability_blue)


def FormTriangleWithRedBoxFn_trial_28(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed to form a triangle with the red box and the cyan box with edge lengths of 20 cm.

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
    next_blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate the distance to form a triangle
    distance_to_red_box = position_norm_metric(next_blue_box_pose, red_box_pose, norm="L2")
    distance_to_cyan_box = position_norm_metric(next_blue_box_pose, cyan_box_pose, norm="L2")
    ideal_distance = 0.20  # 20cm
    # Use linear probability to allow some flexibility around the ideal distance
    probability_to_red_box = linear_probability(distance_to_red_box, 0.15, 0.25)
    probability_to_cyan_box = linear_probability(distance_to_cyan_box, 0.15, 0.25)
    # Combine the probabilities
    total_probability = probability_intersection(probability_to_red_box, probability_to_cyan_box)
    return total_probability


def FormTriangleWithRedAndBlueBoxFn_trial_28(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed to form a triangle with the red box and the blue box with edge lengths of 20 cm.

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
    next_cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate the distance to form a triangle
    distance_to_red_box = position_norm_metric(next_cyan_box_pose, red_box_pose, norm="L2")
    distance_to_blue_box = position_norm_metric(next_cyan_box_pose, blue_box_pose, norm="L2")
    ideal_distance = 0.20  # 20cm
    # Use linear probability to allow some flexibility around the ideal distance
    probability_to_red_box = linear_probability(distance_to_red_box, 0.15, 0.25)
    probability_to_blue_box = linear_probability(distance_to_blue_box, 0.15, 0.25)
    # Combine the probabilities
    total_probability = probability_intersection(probability_to_red_box, probability_to_blue_box)
    return total_probability


def LeftOfRedBoxFn_trial_29(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed left of the red box and aligns with the distance preferences.

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
    # The direction difference should be positive if the object is placed left of the red box.
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the distance to ensure it's within the preferred range
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10  # At least 10cm
    upper_threshold = 1.00  # Ideally 100cm
    distance_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=False)
    # Combine the probabilities
    total_probability = probability_intersection(is_left_probability, distance_probability)
    return total_probability


def InFrontOfRedBoxFn_trial_30(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in front of the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object information from

    Returns:
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
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
    direction_vector = build_direction_vector(red_box_pose, blue_box_pose)
    front_metric = position_diff_along_direction(red_box_pose, blue_box_pose, direction_vector)
    lower_threshold = 0.10  # 10cm as close
    upper_threshold = 0.20  # 20cm as far
    in_front_probability = linear_probability(front_metric, lower_threshold, upper_threshold, is_smaller_then=False)
    return in_front_probability


def BehindCyanBoxFn_trial_30(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed behind the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object information from

    Returns:
        The probability that action `a` on primitive `Place` satisfies the preferences of the human partner.
            Output shape: [batch_size] \in [0, 1].
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
    direction_vector = build_direction_vector(cyan_box_pose, blue_box_pose)
    behind_metric = position_diff_along_direction(cyan_box_pose, blue_box_pose, direction_vector)
    lower_threshold = 0.10  # 10cm as close
    upper_threshold = 0.20  # 20cm as far
    behind_probability = linear_probability(behind_metric, lower_threshold, upper_threshold, is_smaller_then=False)
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
    # Get the screwdriver ID from the environment.
    screwdriver_id = get_object_id_from_name("screwdriver", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the screwdriver, the current state is more reliable.
    screwdriver_pose = get_pose(state, screwdriver_id, -1)
    # Evaluate if the object is placed in a circle of radius 15 cm around the screwdriver
    distance_metric = position_norm_metric(next_object_pose, screwdriver_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.15 - 0.05  # 15cm radius minus tolerance
    upper_threshold = 0.15 + 0.05  # 15cm radius plus tolerance
    circle_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return circle_probability


def FarLeftOfTableFn_trial_32(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed as far to the left of the table as possible.

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
    # Assuming the table's left edge is at x = 0 (normalized), we evaluate how close the object is to this edge.
    # The x-coordinate of the object's pose represents its distance from the left edge.
    x_coordinate = next_object_pose[..., 0]
    # We consider the leftmost 10% of the table as the target area for placing objects.
    # Assuming the table width is 3.0 meters, the target x-coordinate range is [0, 0.3].
    lower_threshold = 0.0  # Left edge of the table
    upper_threshold = 0.3  # 10% into the table width
    # Use linear probability to smoothly transition the preference score as the object moves towards the left edge.
    left_placement_probability = linear_probability(
        x_coordinate, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return left_placement_probability


def PlaceBlueBoxCloseToCyanBoxFn_trial_33(
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
    blue_box_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed close to the cyan box.
    distance_metric = position_norm_metric(blue_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10  # Close but not touching
    upper_threshold = 0.20  # Close enough but not too far
    return linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)


def PlaceRedBoxCloseToBlueAndCyanBoxFn_trial_33(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed close to both the blue and cyan boxes without them touching.

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
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the red box is placed close to the blue box.
    distance_metric_blue = position_norm_metric(red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_cyan = position_norm_metric(red_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10  # Close but not touching
    upper_threshold = 0.20  # Close enough but not too far
    probability_blue = linear_probability(distance_metric_blue, lower_threshold, upper_threshold, is_smaller_then=True)
    probability_cyan = linear_probability(distance_metric_cyan, lower_threshold, upper_threshold, is_smaller_then=True)
    # Combine the probabilities to ensure the red box is close to both the blue and cyan boxes.
    return probability_intersection(probability_blue, probability_cyan)


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
    lower_threshold = torch.pi / 6.0
    upper_threshold = torch.pi / 3.0
    probability = linear_probability(orientation_difference, lower_threshold, upper_threshold, is_smaller_then=True)
    return probability


def InFrontOfCyanBoxFn_trial_35(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in front of the cyan box with a preferred distance.

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
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed in front of the cyan box
    in_front = [1.0, 0.0, 0.0]
    distance_metric = position_diff_along_direction(blue_box_pose, cyan_box_pose, in_front)
    # Using linear probability to evaluate the preferred distance
    return linear_probability(distance_metric, 0.10, 0.20, is_smaller_then=False)


def InFrontOfBlueBoxFn_trial_35(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in front of the blue box with a preferred distance.

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
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    red_box_pose = get_pose(next_state, red_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed in front of the blue box
    in_front = [1.0, 0.0, 0.0]
    distance_metric = position_diff_along_direction(red_box_pose, blue_box_pose, in_front)
    # Using linear probability to evaluate the preferred distance
    return linear_probability(distance_metric, 0.10, 0.20, is_smaller_then=False)


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
    # Calculate the positional difference along the z-axis (alignment).
    alignment_metric = position_diff_along_direction(cyan_box_pose, red_box_pose, direction=[0, 0, 1])
    # Define thresholds for being left and aligned.
    left_threshold = 0.0  # Positive value means left of the red box.
    alignment_threshold = 0.05  # Within 5cm is considered aligned.
    # Calculate probabilities.
    left_probability = threshold_probability(left_of_red_box_metric, left_threshold, is_smaller_then=False)
    alignment_probability = threshold_probability(alignment_metric, alignment_threshold, is_smaller_then=True)
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
    next_red_box_pose = get_pose(next_state, red_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed in front of the blue box.
    distance_metric = position_diff_along_direction(next_red_box_pose, blue_box_pose, direction=[0, 1, 0])
    front_threshold = 0.05  # Minimum distance to consider "in front of"
    front_probability = linear_probability(distance_metric, front_threshold, 0.10, is_smaller_then=False)
    # Evaluate if the red box is aligned with the blue box.
    alignment_metric = position_metric_normal_to_direction(next_red_box_pose, blue_box_pose, direction=[0, 1, 0])
    alignment_probability = linear_probability(alignment_metric, 0.0, 0.05, is_smaller_then=True)
    # Combine probabilities for being in front and aligned.
    total_probability = probability_intersection(front_probability, alignment_probability)
    return total_probability


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
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed left of the red box
    left = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, left)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed left of the red box.
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate if the object is aligned with the red box in the y-axis
    y_diff_metric = position_metric_normal_to_direction(next_object_pose, red_box_pose, left)
    y_diff_probability = linear_probability(y_diff_metric, 0.0, 0.05, is_smaller_then=True)
    # Combine the probabilities
    total_probability = probability_intersection(is_left_probability, y_diff_probability)
    return total_probability


CUSTOM_FNS = {
    "PlaceLeftOfRedBoxAlignedFn_trial_38": PlaceLeftOfRedBoxAlignedFn_trial_38,
    "AlignAndPlaceInFrontOfBlueBoxFn_trial_37": AlignAndPlaceInFrontOfBlueBoxFn_trial_37,
    "LeftOfRedBoxAlignedFn_trial_36": LeftOfRedBoxAlignedFn_trial_36,
    "InFrontOfBlueBoxFn_trial_35": InFrontOfBlueBoxFn_trial_35,
    "InFrontOfCyanBoxFn_trial_35": InFrontOfCyanBoxFn_trial_35,
    "OrientSameAsCyanBoxFn_trial_34": OrientSameAsCyanBoxFn_trial_34,
    "PlaceRedBoxCloseToBlueAndCyanBoxFn_trial_33": PlaceRedBoxCloseToBlueAndCyanBoxFn_trial_33,
    "PlaceBlueBoxCloseToCyanBoxFn_trial_33": PlaceBlueBoxCloseToCyanBoxFn_trial_33,
    "FarLeftOfTableFn_trial_32": FarLeftOfTableFn_trial_32,
    "CircleAroundScrewdriverFn_trial_31": CircleAroundScrewdriverFn_trial_31,
    "BehindCyanBoxFn_trial_30": BehindCyanBoxFn_trial_30,
    "InFrontOfRedBoxFn_trial_30": InFrontOfRedBoxFn_trial_30,
    "LeftOfRedBoxFn_trial_29": LeftOfRedBoxFn_trial_29,
    "FormTriangleWithRedAndBlueBoxFn_trial_28": FormTriangleWithRedAndBlueBoxFn_trial_28,
    "FormTriangleWithRedBoxFn_trial_28": FormTriangleWithRedBoxFn_trial_28,
    "FarFromRedAndBlueBoxFn_trial_27": FarFromRedAndBlueBoxFn_trial_27,
    "CloseToBlueBoxFn_trial_27": CloseToBlueBoxFn_trial_27,
    "AlignBoxesInLineFn_trial_26": AlignBoxesInLineFn_trial_26,
    "PlaceLeftOfRedBoxAlignedFn_trial_25": PlaceLeftOfRedBoxAlignedFn_trial_25,
    "PlaceInFrontOfBlueBoxFn_trial_24": PlaceInFrontOfBlueBoxFn_trial_24,
    "LeftAndAlignedWithRedBoxFn_trial_23": LeftAndAlignedWithRedBoxFn_trial_23,
    "InFrontOfBlueBoxFn_trial_22": InFrontOfBlueBoxFn_trial_22,
    "InFrontOfCyanBoxFn_trial_22": InFrontOfCyanBoxFn_trial_22,
    "OrientSameAsCyanBoxFn_trial_21": OrientSameAsCyanBoxFn_trial_21,
    "PlaceCloseToBlueBoxFn_trial_20": PlaceCloseToBlueBoxFn_trial_20,
    "PlaceCloseToCyanBoxFn_trial_20": PlaceCloseToCyanBoxFn_trial_20,
    "PlaceToLeftOfTableFn_trial_19": PlaceToLeftOfTableFn_trial_19,
    "CircleAroundScrewdriverFn_trial_18": CircleAroundScrewdriverFn_trial_18,
    "PlaceBehindCyanBoxFn_trial_17": PlaceBehindCyanBoxFn_trial_17,
    "PlaceInFrontOfRedBoxFn_trial_17": PlaceInFrontOfRedBoxFn_trial_17,
    "LeftOfRedBoxFn_trial_16": LeftOfRedBoxFn_trial_16,
    "TriangleFormationWithBlueBoxFn_trial_15": TriangleFormationWithBlueBoxFn_trial_15,
    "TriangleFormationWithRedBoxFn_trial_15": TriangleFormationWithRedBoxFn_trial_15,
    "PlaceFarFromRedAndBlueBoxFn_trial_14": PlaceFarFromRedAndBlueBoxFn_trial_14,
    "PlaceCloseToBlueBoxFn_trial_14": PlaceCloseToBlueBoxFn_trial_14,
    "AlignInLineWithBoxesFn_trial_13": AlignInLineWithBoxesFn_trial_13,
    "LeftAndAlignedWithRedBoxFn_trial_12": LeftAndAlignedWithRedBoxFn_trial_12,
    "PlaceInFrontAndAlignedWithBlueBoxFn_trial_11": PlaceInFrontAndAlignedWithBlueBoxFn_trial_11,
    "LeftOfAndAlignedWithRedBoxFn_trial_10": LeftOfAndAlignedWithRedBoxFn_trial_10,
    "LinePlacementRedFrontOfCyanFn_trial_9": LinePlacementRedFrontOfCyanFn_trial_9,
    "LinePlacementBlueFrontOfRedFn_trial_9": LinePlacementBlueFrontOfRedFn_trial_9,
    "OrientLikeCyanAndBlueBoxFn_trial_8": OrientLikeCyanAndBlueBoxFn_trial_8,
    "OrientLikeCyanBoxFn_trial_8": OrientLikeCyanBoxFn_trial_8,
    "PlaceCloseToBlueBoxFn_trial_7": PlaceCloseToBlueBoxFn_trial_7,
    "PlaceCloseToCyanBoxFn_trial_7": PlaceCloseToCyanBoxFn_trial_7,
    "FarLeftOnTableFn_trial_6": FarLeftOnTableFn_trial_6,
    "CircleAroundScrewdriver15cmFn_trial_5": CircleAroundScrewdriver15cmFn_trial_5,
    "PlaceBlueBehindCyanFn_trial_4": PlaceBlueBehindCyanFn_trial_4,
    "PlaceBlueInFrontOfRedFn_trial_4": PlaceBlueInFrontOfRedFn_trial_4,
    "LeftOfRedBoxFn_trial_3": LeftOfRedBoxFn_trial_3,
    "TriangleFormationWithBlueBoxFn_trial_2": TriangleFormationWithBlueBoxFn_trial_2,
    "TriangleFormationWithRedBoxFn_trial_2": TriangleFormationWithRedBoxFn_trial_2,
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
