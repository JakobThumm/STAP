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


def StraightLeftOfRedBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed in a straight line left of the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    # Evaluate the deviation to the front or back
    normal_to_metric = position_metric_normal_to_direction(next_object_pose, red_box_pose, left)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The object should be left of the red box *and* not deviate too much to the front or back.
    total_probability = probability_intersection(is_left_probability, normal_diff_probability)
    return total_probability


def StraightInFrontOfBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed in a straight line in front of the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    is_in_front_of_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the left or right
    normal_to_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, in_front_of)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The object should be in front of the blue box *and* not deviate too much to the left or right.
    total_probability = probability_intersection(is_in_front_of_probability, normal_diff_probability)
    return total_probability


def StraightInFrontOfCyanBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed in a straight line in front of the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the object is placed in front of the cyan box
    in_front_of = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, cyan_box_pose, in_front_of)
    lower_threshold = 0.0
    is_in_front_of_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the left or right
    normal_to_metric = position_metric_normal_to_direction(next_object_pose, cyan_box_pose, in_front_of)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The object should be in front of the cyan box *and* not deviate too much to the left or right.
    total_probability = probability_intersection(is_in_front_of_probability, normal_diff_probability)
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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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


def PlaceNextToRedOrBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed next to the red box or the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed at least 10cm next to the red box.
    distance_red_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10
    upper_threshold = 0.15
    close_to_red_probability = linear_probability(
        distance_red_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Evaluate if the object is placed at least 10cm next to the blue box.
    distance_blue_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10
    upper_threshold = 0.15
    close_to_blue_probability = linear_probability(
        distance_blue_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine the two probabilities with OR logic
    total_probability = probability_union(close_to_red_probability, close_to_blue_probability)
    return total_probability


def StraightRightOfOrBehindBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed in a straight line right of or behind the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    ## Evaluate if the object is placed in a straight line behind the blue box
    # Evaluate if the object is placed behind the blue box
    behind = [-1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, behind)
    lower_threshold = 0.0
    is_behind_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the left or right
    normal_to_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, behind)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The object should be in front of the blue box *and* not deviate too much to the left or right.
    total_straight_front_probability = probability_intersection(is_behind_probability, normal_diff_probability)
    ## Evaluate if the object is placed in a straight line right of the blue box
    # Evaluate if the object is placed right of the blue box
    right = [0.0, 1.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, right)
    lower_threshold = 0.0
    is_right_of_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the front or back
    normal_to_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, right)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The object should be in front of the blue box *and* not deviate too much to the left or right.
    total_straight_right_probability = probability_intersection(is_right_of_probability, normal_diff_probability)
    ## Combine the two probabilities with OR logic
    total_probability = probability_union(total_straight_front_probability, total_straight_right_probability)
    return total_probability


def OrientedSameOrOrthogonalToCyanBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object has the same orientation as the cyan box or is oriented orthogonal to the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Calculate the orientational difference between the object and the cyan box
    orientation_metric = great_circle_distance_metric(next_object_pose, cyan_box_pose)
    lower_threshold = torch.pi / 8.0
    upper_threshold = torch.pi / 6.0
    # Calculate the probability that the object has the same orientation as the cyan box
    same_orientation_probability = linear_probability(
        orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Orthogonal orientation is defined as the object orientation being pi/2 away from the cyan box orientation
    target_value = torch.pi / 2
    # Orthogonal orientation is symmetric around the cyan box orientation
    orthogonal_orientation_metric_1 = orientation_metric - target_value
    orthogonal_orientation_metric_2 = orientation_metric + target_value
    # Calculate the probability that the object has an orthogonal orientation to the cyan box
    orientation_orthogonal_probability_1 = linear_probability(
        orthogonal_orientation_metric_1, lower_threshold, upper_threshold, is_smaller_then=True
    )
    orientation_orthogonal_probability_2 = linear_probability(
        orthogonal_orientation_metric_2, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine the two orthogonal orientation probabilities with OR logic
    orientation_orthogonal_probability = probability_union(
        orientation_orthogonal_probability_1, orientation_orthogonal_probability_2
    )
    # Combine the two probabilities with OR logic
    total_probability = probability_union(same_orientation_probability, orientation_orthogonal_probability)
    return total_probability


def OrientedSameAsCyanOrRedBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the if the object is oriented the same as the cyan box or the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    lower_threshold = torch.pi / 8.0
    upper_threshold = torch.pi / 6.0
    # Evaluate if the object has the same orientation as the cyan box
    cyan_orientation_metric = great_circle_distance_metric(next_object_pose, cyan_box_pose)
    cyan_orientation_probability = linear_probability(
        cyan_orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Evaluate if the object has the same orientation as the red box
    red_orientation_metric = great_circle_distance_metric(next_object_pose, red_box_pose)
    red_orientation_probability = linear_probability(
        red_orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # combine the two probabilities with OR logic
    total_probability = probability_union(cyan_orientation_probability, red_orientation_probability)
    return total_probability


def PlaceNextToOrOrientedSameAsRedBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the if the object is placed next to or oriented the same as the the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed next to the red box.
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10
    upper_threshold = 0.15
    next_to_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    # Evaluate if the object has the same orientation as the red box
    lower_threshold = torch.pi / 8.0
    upper_threshold = torch.pi / 6.0
    orientation_metric = great_circle_distance_metric(next_object_pose, red_box_pose)
    orientation_probability = linear_probability(
        orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # combine the two probabilities with OR logic
    total_probability = probability_union(next_to_probability, orientation_probability)
    return total_probability


def InLineWithRedAndBlueBoxFn_trial_0(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed in line with the red and blue boxes.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate the direction vector from the red box to the blue box
    direction_vector_red_to_blue = build_direction_vector(red_box_pose, blue_box_pose)
    # Calculate the positional difference of the cyan box normal to the direction from red to blue
    normal_to_red_blue_line_metric = position_metric_normal_to_direction(
        cyan_box_pose, red_box_pose, direction_vector_red_to_blue
    )
    # Evaluate if the cyan box is placed in line with the red and blue boxes
    lower_threshold = 0.0
    upper_threshold = 0.05
    in_line_probability = linear_probability(
        normal_to_red_blue_line_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return in_line_probability


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
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    red_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_red_box_pose = get_pose(next_state, red_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed close to the blue box.
    distance_metric = position_norm_metric(next_red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.05  # 5cm
    upper_threshold = 0.10  # 10cm
    close_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return close_probability


def FarFromRedAndBlueBoxFn_trial_1(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed far away from both the red and the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    # Evaluate if the cyan box is placed far away from both the red and the blue box.
    distance_to_red_metric = position_norm_metric(next_cyan_box_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_to_blue_metric = position_norm_metric(next_cyan_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.20  # 20cm
    upper_threshold = 1.00  # 100cm
    far_from_red_probability = linear_probability(
        distance_to_red_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    far_from_blue_probability = linear_probability(
        distance_to_blue_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine probabilities to ensure the cyan box is far from both boxes.
    return probability_intersection(far_from_red_probability, far_from_blue_probability)


def TriangleFormationFn_trial_2(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the objects are arranged in a triangle formation with each edge being 20 cm.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, get_object_id_from_primitive(0, primitive), -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Calculate distances between each pair of boxes.
    distance_red_blue = position_norm_metric(red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    distance_red_cyan = position_norm_metric(red_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    distance_blue_cyan = position_norm_metric(blue_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    # Define the target distance and tolerance for the triangle formation.
    target_distance = 0.20  # 20 cm
    tolerance = 0.02  # 2 cm tolerance
    # Calculate the probability for each distance to be within the target range.
    prob_red_blue = linear_probability(
        distance_red_blue, target_distance - tolerance, target_distance + tolerance, is_smaller_then=False
    )
    prob_red_cyan = linear_probability(
        distance_red_cyan, target_distance - tolerance, target_distance + tolerance, is_smaller_then=False
    )
    prob_blue_cyan = linear_probability(
        distance_blue_cyan, target_distance - tolerance, target_distance + tolerance, is_smaller_then=False
    )
    # Combine the probabilities to ensure all distances meet the criteria.
    total_probability = probability_intersection(prob_red_blue, probability_intersection(prob_red_cyan, prob_blue_cyan))
    return total_probability


def LeftOfRedBoxFn_trial_3(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed to the left of the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed to the left of the red box
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, [0, 1, 0])
    lower_threshold = 0.0
    left_of_red_box_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return left_of_red_box_probability


def CircleAroundScrewdriverFn_trial_4(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed in a circle of radius 15 cm around the screwdriver.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    screwdriver_id = get_object_id_from_name("screwdriver", env, primitive)
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    screwdriver_pose = get_pose(state, screwdriver_id, -1)
    # Calculate the distance between the object and the screwdriver
    distance_metric = position_norm_metric(next_object_pose, screwdriver_pose, norm="L2", axes=["x", "y"])
    # The desired radius is 15 cm
    lower_threshold = 0.15 - 0.05  # Allowing a small margin
    upper_threshold = 0.15 + 0.05  # Allowing a small margin
    # Calculate the probability that the object is within the desired distance range from the screwdriver
    circle_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return circle_probability


def PlaceCloseButNotTouchingFn_trial_5(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed close to the other boxes but not touching.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate distances to other boxes
    distance_to_cyan = position_norm_metric(next_object_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    distance_to_red = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_to_blue = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    # Define thresholds for "close but not touching"
    lower_threshold = 0.05  # Minimum distance to be considered not touching
    upper_threshold = 0.10  # Maximum distance to be considered close
    # Calculate probabilities
    prob_cyan = linear_probability(distance_to_cyan, lower_threshold, upper_threshold, is_smaller_then=True)
    prob_red = linear_probability(distance_to_red, lower_threshold, upper_threshold, is_smaller_then=True)
    prob_blue = linear_probability(distance_to_blue, lower_threshold, upper_threshold, is_smaller_then=True)
    # Combine probabilities
    total_probability = probability_intersection(prob_cyan, probability_intersection(prob_red, prob_blue))
    return total_probability


def OrientedSameAsCyanBoxFn_trial_6(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object has the same orientation as the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Calculate the orientational difference between the object and the cyan box
    orientation_metric = great_circle_distance_metric(next_object_pose, cyan_box_pose)
    lower_threshold = torch.pi / 8.0
    upper_threshold = torch.pi / 6.0
    # Calculate the probability that the object has the same orientation as the cyan box
    same_orientation_probability = linear_probability(
        orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return same_orientation_probability


def PlaceBlueBoxInLineFn_trial_7(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in front of the cyan box in a straight line.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    next_blue_box_pose = get_pose(next_state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Calculate the positional difference along the front/behind direction ([+/-1, 0, 0])
    position_diff = position_diff_along_direction(next_blue_box_pose, cyan_box_pose, direction=[1, 0, 0])
    # Calculate the positional difference normal to the front/behind direction to ensure straight line
    position_diff_normal = position_metric_normal_to_direction(next_blue_box_pose, cyan_box_pose, direction=[1, 0, 0])
    # Define thresholds for being in a straight line and in front
    in_line_threshold = 0.05  # 5cm tolerance for being in a straight line
    in_front_threshold = 0.10  # At least 10cm in front
    # Calculate probabilities
    in_line_probability = threshold_probability(position_diff_normal, in_line_threshold, is_smaller_then=True)
    in_front_probability = threshold_probability(position_diff, in_front_threshold, is_smaller_then=False)
    # Combine probabilities
    total_probability = probability_intersection(in_line_probability, in_front_probability)
    return total_probability


def PlaceRedBoxInLineFn_trial_7(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in front of the blue box in a straight line.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    red_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_red_box_pose = get_pose(next_state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate the positional difference along the front/behind direction ([+/-1, 0, 0])
    position_diff = position_diff_along_direction(next_red_box_pose, blue_box_pose, direction=[1, 0, 0])
    # Calculate the positional difference normal to the front/behind direction to ensure straight line
    position_diff_normal = position_metric_normal_to_direction(next_red_box_pose, blue_box_pose, direction=[1, 0, 0])
    # Define thresholds for being in a straight line and in front
    in_line_threshold = 0.05  # 5cm tolerance for being in a straight line
    in_front_threshold = 0.10  # At least 10cm in front
    # Calculate probabilities
    in_line_probability = threshold_probability(position_diff_normal, in_line_threshold, is_smaller_then=True)
    in_front_probability = threshold_probability(position_diff, in_front_threshold, is_smaller_then=False)
    # Combine probabilities
    total_probability = probability_intersection(in_line_probability, in_front_probability)
    return total_probability


def StraightLineLeftOfRedBoxFn_trial_8(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in a straight line left of the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the cyan box is placed to the left of the red box
    to_the_left_of = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(next_cyan_box_pose, red_box_pose, to_the_left_of)
    lower_threshold = 0.0
    is_to_the_left_of_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the front or back
    normal_to_metric = position_metric_normal_to_direction(next_cyan_box_pose, red_box_pose, to_the_left_of)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The cyan box should be to the left of the red box *and* not deviate too much to the front or back.
    total_probability = probability_intersection(is_to_the_left_of_probability, normal_diff_probability)
    return total_probability


def StraightInFrontOfBlueBoxFn_trial_9(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in a straight line in front of the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    is_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the left or right
    normal_to_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, front)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The object should be in front of the blue box *and* not deviate too much to the left or right.
    total_probability = probability_intersection(is_front_probability, normal_diff_probability)
    return total_probability


def CloseToRedOrBlueBoxFn_trial_10(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed close to either the red box or the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate distance to red box
    distance_to_red = position_norm_metric(next_cyan_box_pose, red_box_pose, norm="L2")
    # Calculate distance to blue box
    distance_to_blue = position_norm_metric(next_cyan_box_pose, blue_box_pose, norm="L2")
    # Define thresholds for "close"
    lower_threshold = 0.0  # At least 0m away to be considered close
    upper_threshold = 0.1  # Up to 10cm away to be considered close
    # Calculate probabilities
    probability_close_to_red = linear_probability(
        distance_to_red, lower_threshold, upper_threshold, is_smaller_then=True
    )
    probability_close_to_blue = linear_probability(
        distance_to_blue, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine probabilities using OR logic
    total_probability = probability_union(probability_close_to_red, probability_close_to_blue)
    return total_probability


def PlaceInLineRightOrBehindBlueBoxFn_trial_11(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the red box is placed in a straight line right of or behind the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    # Evaluate if the red box is placed right of the blue box
    right_of_blue = position_diff_along_direction(red_box_pose, blue_box_pose, direction=[0, 1, 0])
    # Evaluate if the red box is placed behind the blue box
    behind_blue = position_diff_along_direction(red_box_pose, blue_box_pose, direction=[-1, 0, 0])
    # Combine the probabilities
    right_of_blue_probability = threshold_probability(right_of_blue, 0.0, is_smaller_then=False)
    behind_blue_probability = threshold_probability(behind_blue, 0.0, is_smaller_then=False)
    total_probability = probability_union(right_of_blue_probability, behind_blue_probability)
    return total_probability


def OrientLikeOrOrthogonalToCyanBoxFn_trial_12(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is oriented like or orthogonal to the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    next_object_orientation = get_pose(next_state, object_id, -1)[..., 3:]
    cyan_box_orientation = get_pose(state, cyan_box_id, -1)[..., 3:]
    # Calculate the orientation difference
    orientation_diff = great_circle_distance_metric(next_object_orientation, cyan_box_orientation)
    # Calculate the probability of being oriented like or orthogonal
    like_orientation_probability = linear_probability(orientation_diff, 0, torch.pi / 4, is_smaller_then=True)
    orthogonal_orientation_probability = linear_probability(
        orientation_diff, torch.pi / 4, 3 * torch.pi / 4, is_smaller_then=True
    )
    total_probability = probability_union(like_orientation_probability, orthogonal_orientation_probability)
    return total_probability


def OrientLikeCyanOrRedBoxFn_trial_13(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is oriented the same as the cyan box or the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    # Evaluate if the blue box has the same orientation as the cyan box
    orientation_metric_cyan = great_circle_distance_metric(blue_box_pose, cyan_box_pose)
    orientation_metric_red = great_circle_distance_metric(blue_box_pose, red_box_pose)
    lower_threshold = torch.pi / 8.0
    upper_threshold = torch.pi / 6.0
    orientation_probability_cyan = linear_probability(
        orientation_metric_cyan, lower_threshold, upper_threshold, is_smaller_then=True
    )
    orientation_probability_red = linear_probability(
        orientation_metric_red, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine the two probabilities with OR logic
    total_probability = probability_union(orientation_probability_cyan, orientation_probability_red)
    return total_probability


def CloseOrOrientedSameAsRedBoxFn_trial_14(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is close to or oriented the same as the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Distance metric for closeness
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    close_threshold = 0.10  # 10cm as close threshold
    close_probability = threshold_probability(distance_metric, close_threshold, is_smaller_then=True)
    # Orientation metric for similarity
    orientation_metric = great_circle_distance_metric(next_object_pose, red_box_pose)
    orientation_threshold = torch.pi / 8.0  # Small rotation difference threshold
    orientation_probability = threshold_probability(orientation_metric, orientation_threshold, is_smaller_then=True)
    # Combine probabilities using OR logic (union)
    total_probability = probability_union(close_probability, orientation_probability)
    return total_probability


def AlignInLineWithBoxesFn_trial_15(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in line with the red and blue boxes.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate the direction vector from red to blue box
    direction_red_to_blue = build_direction_vector(red_box_pose, blue_box_pose)
    # Calculate the normal distance of the cyan box to the line formed by red and blue boxes
    normal_distance_to_line = position_metric_normal_to_direction(cyan_box_pose, red_box_pose, direction_red_to_blue)
    # Calculate the distance along the direction from red to blue box to ensure it's between them
    distance_along_line = position_diff_along_direction(cyan_box_pose, red_box_pose, direction_red_to_blue)
    distance_blue_to_red = position_diff_along_direction(blue_box_pose, red_box_pose, direction_red_to_blue)
    # Probability calculations
    normal_distance_probability = linear_probability(normal_distance_to_line, 0.0, 0.05, is_smaller_then=True)
    between_boxes_probability = linear_probability(distance_along_line, 0.0, distance_blue_to_red, is_smaller_then=True)
    # Combine probabilities to ensure the cyan box is in line and between the red and blue boxes
    total_probability = probability_intersection(normal_distance_probability, between_boxes_probability)
    return total_probability


def CloseToBlueBoxFn_trial_16(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the red box is placed close to the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    distance_metric = position_norm_metric(red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.05  # 5cm
    upper_threshold = 0.10  # 10cm
    close_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return close_probability


def FarFromRedAndBlueBoxFn_trial_16(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed far away from both the red and the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    distance_to_red = position_norm_metric(cyan_box_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_to_blue = position_norm_metric(cyan_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.20  # 20cm
    upper_threshold = 1.00  # 100cm
    far_from_red_probability = linear_probability(
        distance_to_red, lower_threshold, upper_threshold, is_smaller_then=True
    )
    far_from_blue_probability = linear_probability(
        distance_to_blue, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine probabilities to ensure the cyan box is far from both boxes
    return probability_intersection(far_from_red_probability, far_from_blue_probability)


def TrianglePlacementBlueBoxFn_trial_17(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the blue box is placed forming a triangle with the red and cyan boxes.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    # Evaluate if the object is placed forming a triangle
    distance_metric_red = position_norm_metric(blue_box_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_cyan = position_norm_metric(blue_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    triangle_edge_length = 0.20
    probability_red = linear_probability(
        distance_metric_red, triangle_edge_length - 0.05, triangle_edge_length + 0.05, is_smaller_then=True
    )
    probability_cyan = linear_probability(
        distance_metric_cyan, triangle_edge_length - 0.05, triangle_edge_length + 0.05, is_smaller_then=True
    )
    return probability_intersection(probability_red, probability_cyan)


def TrianglePlacementCyanBoxFn_trial_17(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed forming a triangle with the red and blue boxes.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    # Evaluate if the object is placed forming a triangle
    distance_metric_red = position_norm_metric(cyan_box_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_metric_blue = position_norm_metric(cyan_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    triangle_edge_length = 0.20
    probability_red = linear_probability(
        distance_metric_red, triangle_edge_length - 0.05, triangle_edge_length + 0.05, is_smaller_then=True
    )
    probability_blue = linear_probability(
        distance_metric_blue, triangle_edge_length - 0.05, triangle_edge_length + 0.05, is_smaller_then=True
    )
    return probability_intersection(probability_red, probability_blue)


def LeftOfRedBoxFn_trial_18(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed left of the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object ID from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated object, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed left of the red box
    direction = [0, 1, 0]  # Positive Y direction indicates left in our coordinate system
    positional_difference = position_diff_along_direction(next_object_pose, red_box_pose, direction)
    # The positional difference should be positive, indicating it is to the left.
    lower_threshold = 0.0  # At least 0 to ensure it's to the left
    upper_threshold = 0.2  # Arbitrary upper bound to ensure closeness
    probability = linear_probability(positional_difference, lower_threshold, upper_threshold, is_smaller_then=False)
    return probability


def CircleAroundScrewdriverFn_trial_19(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed in a circle of radius 15 cm around the screwdriver.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    # Evaluate if the object is placed in a circle of radius 15 cm around the screwdriver.
    distance_metric = position_norm_metric(next_object_pose, screwdriver_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.15 - 0.05  # 15cm radius minus tolerance
    upper_threshold = 0.15 + 0.05  # 15cm radius plus tolerance
    circle_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return circle_probability


def PlaceCloseToOthersFn_trial_20(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed close to the other two boxes without them touching.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)

    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)

    # Calculate distances to other boxes
    distance_to_cyan = position_norm_metric(next_object_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    distance_to_red = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    distance_to_blue = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])

    # Define thresholds for "close" and "not touching"
    lower_threshold = 0.05  # Minimum distance to consider "not touching"
    upper_threshold = 0.10  # Maximum distance to consider "close"

    # Calculate probabilities
    prob_cyan = linear_probability(distance_to_cyan, lower_threshold, upper_threshold, is_smaller_then=True)
    prob_red = linear_probability(distance_to_red, lower_threshold, upper_threshold, is_smaller_then=True)
    prob_blue = linear_probability(distance_to_blue, lower_threshold, upper_threshold, is_smaller_then=True)

    # Combine probabilities
    total_probability = probability_intersection(prob_cyan, probability_intersection(prob_red, prob_blue))
    return total_probability


def OrientedSameAsCyanBoxFn_trial_21(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is oriented the same as the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    lower_threshold = torch.pi / 8.0
    upper_threshold = torch.pi / 6.0
    orientation_metric = great_circle_distance_metric(next_object_pose, cyan_box_pose)
    orientation_probability = linear_probability(
        orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return orientation_probability


def BlueInFrontOfCyanFn_trial_22(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in front of the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed in front of the cyan box
    front_direction = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(blue_box_pose, cyan_box_pose, front_direction)
    lower_threshold = 0.0
    is_in_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the left or right
    normal_to_metric = position_metric_normal_to_direction(blue_box_pose, cyan_box_pose, front_direction)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The blue box should be in front of the cyan box *and* not deviate too much to the left or right.
    total_probability = probability_intersection(is_in_front_probability, normal_diff_probability)
    return total_probability


def RedInFrontOfBlueFn_trial_22(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in front of the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    red_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_pose = get_pose(next_state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed in front of the blue box
    front_direction = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(red_box_pose, blue_box_pose, front_direction)
    lower_threshold = 0.0
    is_in_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the left or right
    normal_to_metric = position_metric_normal_to_direction(red_box_pose, blue_box_pose, front_direction)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The red box should be in front of the blue box *and* not deviate too much to the left or right.
    total_probability = probability_intersection(is_in_front_probability, normal_diff_probability)
    return total_probability


def StraightLineLeftOfRedBoxFn_trial_23(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in a straight line left of the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_cyan_box_pose = get_pose(next_state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the cyan box is placed to the left of the red box
    to_the_left_of = [0, -1.0, 0]
    direction_difference = position_diff_along_direction(next_cyan_box_pose, red_box_pose, to_the_left_of)
    lower_threshold = 0.0
    is_to_the_left_of_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the front or back
    normal_to_metric = position_metric_normal_to_direction(next_cyan_box_pose, red_box_pose, to_the_left_of)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The cyan box should be to the left of the red box *and* not deviate too much to the front or back.
    total_probability = probability_intersection(is_to_the_left_of_probability, normal_diff_probability)
    return total_probability


def StraightInFrontOfBlueBoxFn_trial_24(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in a straight line in front of the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    is_in_front_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the right or left
    normal_to_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, front)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The object should be in front of the blue box *and* not deviate too much to the right or left.
    total_probability = probability_intersection(is_in_front_probability, normal_diff_probability)
    return total_probability


def CloseToRedOrBlueBoxFn_trial_25(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed close to either the red box or the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    # Evaluate if the cyan box is placed close to the red box
    distance_to_red_box = position_norm_metric(cyan_box_pose, red_box_pose, norm="L2", axes=["x", "y"])
    # Evaluate if the cyan box is placed close to the blue box
    distance_to_blue_box = position_norm_metric(cyan_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    # Define thresholds for "close"
    lower_threshold = 0.0
    upper_threshold = 0.10
    # Calculate probabilities
    close_to_red_box_probability = linear_probability(
        distance_to_red_box, lower_threshold, upper_threshold, is_smaller_then=True
    )
    close_to_blue_box_probability = linear_probability(
        distance_to_blue_box, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The cyan box should be close to either the red box or the blue box.
    total_probability = probability_union(close_to_red_box_probability, close_to_blue_box_probability)
    return total_probability


def PlaceRightOrBehindBlueBoxFn_trial_26(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in a straight line right of or behind the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate the positional difference along the right (y-axis) and behind (x-axis) directions
    diff_right = position_diff_along_direction(next_object_pose, blue_box_pose, direction=[0, 1, 0])
    diff_behind = position_diff_along_direction(next_object_pose, blue_box_pose, direction=[-1, 0, 0])
    # Evaluate if the object is placed right of or behind the blue box
    right_probability = threshold_probability(diff_right, 0.0, is_smaller_then=False)
    behind_probability = threshold_probability(diff_behind, 0.0, is_smaller_then=False)
    # Combine the probabilities with OR logic
    total_probability = probability_union(right_probability, behind_probability)
    return total_probability


def OrientedSameOrOrthogonalToCyanBoxFn_trial_27(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is oriented the same as the cyan box or orthogonal to it.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the object has the same orientation as the cyan box
    same_orientation_metric = great_circle_distance_metric(next_object_pose, cyan_box_pose)
    same_orientation_probability = linear_probability(
        same_orientation_metric, torch.pi / 8.0, torch.pi / 6.0, is_smaller_then=True
    )
    # Evaluate if the object is orthogonal to the cyan box
    orthogonal_orientation_metric = torch.abs(same_orientation_metric - torch.pi / 2)
    orthogonal_orientation_probability = linear_probability(
        orthogonal_orientation_metric, torch.pi / 8.0, torch.pi / 6.0, is_smaller_then=True
    )
    # Combine the probabilities with OR logic
    total_probability = probability_union(same_orientation_probability, orthogonal_orientation_probability)
    return total_probability


def OrientationSameAsCyanOrRedBoxFn_trial_28(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is oriented the same as the cyan box or the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        The probability that the blue box's orientation satisfies the preferences of the human partner [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)

    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)

    # Calculate orientation difference with cyan box
    orientation_diff_cyan = great_circle_distance_metric(blue_box_pose, cyan_box_pose)
    # Calculate orientation difference with red box
    orientation_diff_red = great_circle_distance_metric(blue_box_pose, red_box_pose)

    # Define thresholds for considering orientations "the same"
    lower_threshold = 0.0  # No difference
    upper_threshold = torch.pi / 8  # Small difference

    # Calculate probabilities
    probability_cyan = linear_probability(orientation_diff_cyan, lower_threshold, upper_threshold, is_smaller_then=True)
    probability_red = linear_probability(orientation_diff_red, lower_threshold, upper_threshold, is_smaller_then=True)

    # The blue box should be oriented the same as either the cyan box or the red box
    total_probability = probability_union(probability_cyan, probability_red)

    return total_probability


def CloseOrOrientedSameAsRedBoxFn_trial_29(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed close to or oriented the same as the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed close to the red box
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    close_distance_probability = linear_probability(distance_metric, 0.05, 0.10, is_smaller_then=True)
    # Evaluate if the object is oriented the same as the red box
    orientation_metric = great_circle_distance_metric(next_object_pose, red_box_pose)
    orientation_probability = linear_probability(
        orientation_metric, torch.pi / 8.0, torch.pi / 6.0, is_smaller_then=True
    )
    # Combine the probabilities with OR logic
    total_probability = probability_union(close_distance_probability, orientation_probability)
    return total_probability


def AlignInLineOnTableFn_trial_30(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed in line with the red and blue boxes on the table.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    # Evaluate if the cyan box is placed in line with the red and blue boxes
    direction_vector_red_to_blue = build_direction_vector(red_box_pose, blue_box_pose)
    direction_vector_red_to_cyan = build_direction_vector(red_box_pose, cyan_box_pose)
    direction_vector_blue_to_cyan = build_direction_vector(blue_box_pose, cyan_box_pose)
    # Calculate the angle between the direction vectors to ensure they are aligned
    angle_red_to_blue_cyan = great_circle_distance_metric(direction_vector_red_to_blue, direction_vector_red_to_cyan)
    angle_blue_to_red_cyan = great_circle_distance_metric(direction_vector_red_to_blue, direction_vector_blue_to_cyan)
    # Define thresholds for alignment
    lower_threshold = 0.0
    upper_threshold = torch.pi / 8.0  # Small angle deviation allowed for alignment
    # Calculate probabilities
    alignment_probability_red_cyan = linear_probability(
        angle_red_to_blue_cyan, lower_threshold, upper_threshold, is_smaller_then=True
    )
    alignment_probability_blue_cyan = linear_probability(
        angle_blue_to_red_cyan, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine probabilities to ensure both conditions are met
    total_probability = probability_intersection(alignment_probability_red_cyan, alignment_probability_blue_cyan)
    return total_probability


def CloseToBlueBoxFn_trial_31(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the red box is placed close to the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    red_box_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_red_box_pose = get_pose(next_state, red_box_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed close to the blue box
    distance_metric = position_norm_metric(next_red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10
    upper_threshold = 0.20
    closeness_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return closeness_probability


def FarFromRedAndBlueBoxFn_trial_31(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the cyan box is placed far away from both the red and the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    # Evaluate if the cyan box is placed far away from the red box
    distance_metric_red = position_norm_metric(next_cyan_box_pose, red_box_pose, norm="L2", axes=["x", "y"])
    # Evaluate if the cyan box is placed far away from the blue box
    distance_metric_blue = position_norm_metric(next_cyan_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.20
    upper_threshold = 1.00
    far_probability_red = linear_probability(
        distance_metric_red, lower_threshold, upper_threshold, is_smaller_then=False
    )
    far_probability_blue = linear_probability(
        distance_metric_blue, lower_threshold, upper_threshold, is_smaller_then=False
    )
    # Combine the probabilities to ensure the cyan box is far from both the red and blue boxes
    total_far_probability = probability_intersection(far_probability_red, far_probability_blue)
    return total_far_probability


def TriangleFormationFn_trial_32(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the boxes are arranged in a triangle formation with each edge being 20 cm.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object information from

    Returns:
        The probability that action `a` on primitive `Place` satisfies the triangle formation preference.
            Output shape: [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get object IDs
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    # Get poses
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Calculate distances
    red_blue_distance = position_norm_metric(red_box_pose, blue_box_pose, "L2")
    red_cyan_distance = position_norm_metric(red_box_pose, cyan_box_pose, "L2")
    blue_cyan_distance = position_norm_metric(blue_box_pose, cyan_box_pose, "L2")
    # Desired distance in meters
    desired_distance = 0.2  # 20 cm
    # Calculate probabilities
    red_blue_prob = linear_probability(red_blue_distance, desired_distance - 0.01, desired_distance + 0.01)
    red_cyan_prob = linear_probability(red_cyan_distance, desired_distance - 0.01, desired_distance + 0.01)
    blue_cyan_prob = linear_probability(blue_cyan_distance, desired_distance - 0.01, desired_distance + 0.01)
    # Combine probabilities
    total_probability = probability_intersection(probability_intersection(red_blue_prob, red_cyan_prob), blue_cyan_prob)
    return total_probability


def PlaceLeftOfRedBoxFn_trial_33(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed left of the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    direction = [0, 1, 0]  # Left direction in the y-axis
    position_diff_metric = position_diff_along_direction(next_object_pose, red_box_pose, direction)
    lower_threshold = 0.0  # At least 0 to be considered left
    upper_threshold = 0.2  # Up to 20cm to the left is strongly preferred
    probability = linear_probability(position_diff_metric, lower_threshold, upper_threshold, is_smaller_then=False)
    return probability


def PlaceInCircleAroundScrewdriver15cmFn_trial_34(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluate if the object is placed in a circle of radius 15cm around the screwdriver.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    lower_threshold = 0.14
    ideal_point = 0.15
    upper_threshold = 0.16
    smaller_than_ideal_probability = linear_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability(
        distance_metric, ideal_point, upper_threshold, is_smaller_then=True
    )
    return probability_intersection(smaller_than_ideal_probability, bigger_than_ideal_probability)


def PlaceCloseWithoutTouchingFn_trial_35(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is placed close to the other boxes without them touching.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    object_ids = [cyan_box_id, red_box_id, blue_box_id]
    probabilities = []

    for object_id in object_ids:
        if object_id == get_object_id_from_primitive(0, primitive):
            continue  # Skip the object being placed
        next_object_pose = get_pose(next_state, get_object_id_from_primitive(0, primitive), -1)
        other_object_pose = get_pose(state, object_id, -1)
        distance_metric = position_norm_metric(next_object_pose, other_object_pose, norm="L2", axes=["x", "y"])
        # Define thresholds for being close but not touching
        lower_threshold = 0.10  # Close enough
        upper_threshold = 0.05  # Not touching
        probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=False)
        probabilities.append(probability)

    # Combine probabilities for being close to any of the other boxes without touching
    if probabilities:
        total_probability = probabilities[0]
        for prob in probabilities[1:]:
            total_probability = probability_union(total_probability, prob)
    else:
        total_probability = torch.tensor(1.0)  # Default to 1 if no other boxes to compare to

    return total_probability


def OrientedSameAsCyanBoxFn_trial_36(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is oriented the same as the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    lower_threshold = torch.pi / 8.0
    upper_threshold = torch.pi / 6.0
    # Evaluate if the object has the same orientation as the cyan box
    orientation_metric = great_circle_distance_metric(next_object_pose, cyan_box_pose)
    orientation_probability = linear_probability(
        orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return orientation_probability


def StraightLineCyanToFrontFn_trial_37(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed in a straight line in front of the cyan box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    next_blue_box_pose = get_pose(next_state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the blue box is placed in front of the cyan box
    in_front_of = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_blue_box_pose, cyan_box_pose, in_front_of)
    lower_threshold = 0.0
    is_in_front_of_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the left or right
    normal_to_metric = position_metric_normal_to_direction(next_blue_box_pose, cyan_box_pose, in_front_of)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The blue box should be in front of the cyan box *and* not deviate too much to the left or right.
    total_probability = probability_intersection(is_in_front_of_probability, normal_diff_probability)
    return total_probability


def StraightLineBlueToFrontFn_trial_37(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in a straight line in front of the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    red_box_id = get_object_id_from_primitive(0, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_red_box_pose = get_pose(next_state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the red box is placed in front of the blue box
    in_front_of = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_red_box_pose, blue_box_pose, in_front_of)
    lower_threshold = 0.0
    is_in_front_of_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the left or right
    normal_to_metric = position_metric_normal_to_direction(next_red_box_pose, blue_box_pose, in_front_of)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The red box should be in front of the blue box *and* not deviate too much to the left or right.
    total_probability = probability_intersection(is_in_front_of_probability, normal_diff_probability)
    return total_probability


def StraightLineLeftOfRedBoxFn_trial_38(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed in a straight line left of the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed to the left of the red box
    to_the_left_of = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, to_the_left_of)
    lower_threshold = 0.0
    is_to_the_left_of_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the front or back
    normal_to_metric = position_metric_normal_to_direction(next_object_pose, red_box_pose, to_the_left_of)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The object should be to the left of the red box *and* not deviate too much to the front or back.
    total_probability = probability_intersection(is_to_the_left_of_probability, normal_diff_probability)
    return total_probability


def StraightInFrontOfBlueBoxFn_trial_39(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in a straight line in front of the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    is_in_front_of_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the left or right
    normal_to_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, in_front_of)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The object should be in front of the blue box *and* not deviate too much to the left or right.
    total_probability = probability_intersection(is_in_front_of_probability, normal_diff_probability)
    return total_probability


def PlaceCloseToRedOrBlueBoxFn_trial_40(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the cyan box is placed close to either the red box or the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
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
    # Evaluate if the cyan box is placed close to the red box.
    distance_metric_red = position_norm_metric(cyan_box_pose, red_box_pose, norm="L2", axes=["x", "y"])
    # Evaluate if the cyan box is placed close to the blue box.
    distance_metric_blue = position_norm_metric(cyan_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    # Define thresholds for "close" proximity.
    lower_threshold = 0.05  # 5cm is considered close.
    upper_threshold = 0.10  # Up to 10cm is still acceptable.
    # Calculate probabilities for being close to each box.
    close_to_red_probability = linear_probability(
        distance_metric_red, lower_threshold, upper_threshold, is_smaller_then=True
    )
    close_to_blue_probability = linear_probability(
        distance_metric_blue, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine the probabilities using OR logic, since being close to either box satisfies the condition.
    total_probability = probability_union(close_to_red_probability, close_to_blue_probability)
    return total_probability


def PlaceInLineRightOrBehindBlueBoxFn_trial_41(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the red box is placed in a straight line right of or behind the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_red_box_pose = get_pose(next_state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Calculate the positional difference along the right (y-axis) and behind (x-axis) directions
    right_diff = position_diff_along_direction(next_red_box_pose, blue_box_pose, direction=[0, 1, 0])
    behind_diff = position_diff_along_direction(next_red_box_pose, blue_box_pose, direction=[1, 0, 0])
    # Define thresholds for being considered "in line"
    lower_threshold = 0.0  # Positive values indicate right of or behind
    upper_threshold = 0.05  # Small tolerance for alignment
    right_probability = linear_probability(right_diff, lower_threshold, upper_threshold, is_smaller_then=False)
    behind_probability = linear_probability(behind_diff, lower_threshold, upper_threshold, is_smaller_then=False)
    # Combine probabilities with OR logic since it can be either right or behind
    total_probability = probability_union(right_probability, behind_probability)
    return total_probability


def OrientLikeCyanBoxFn_trial_42(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the object is oriented in the same direction as the cyan box or orthogonal to it.

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
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    next_object_orientation = get_pose(next_state, object_id, -1)[..., 3:]
    cyan_box_orientation = get_pose(state, cyan_box_id, -1)[..., 3:]
    # Calculate the great circle distance between the orientations
    orientation_similarity = great_circle_distance_metric(next_object_orientation, cyan_box_orientation)
    # Calculate the angle for orthogonality (90 degrees or pi/2 radians)
    orthogonal_threshold = torch.pi / 2
    # Check if the orientation is similar (small angle) or orthogonal (close to 90 degrees)
    similarity_probability = linear_probability(orientation_similarity, 0, torch.pi / 6, is_smaller_then=True)
    orthogonality_probability = linear_probability(
        torch.abs(orientation_similarity - orthogonal_threshold), 0, torch.pi / 6, is_smaller_then=True
    )
    # The object should be either similar or orthogonal in orientation to the cyan box.
    total_probability = probability_union(similarity_probability, orthogonality_probability)
    return total_probability


def OrientSameAsCyanOrRedBoxFn_trial_43(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is oriented the same as the cyan box or the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_primitive(0, primitive)
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_pose = get_pose(next_state, blue_box_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)

    # Calculate orientation difference with cyan box
    orientation_diff_cyan = great_circle_distance_metric(blue_box_pose, cyan_box_pose)
    # Calculate orientation difference with red box
    orientation_diff_red = great_circle_distance_metric(blue_box_pose, red_box_pose)

    # Define thresholds for considering orientations "the same"
    lower_threshold = 0.0  # No difference
    upper_threshold = torch.pi / 8  # Small difference

    # Calculate probabilities
    probability_cyan = linear_probability(orientation_diff_cyan, lower_threshold, upper_threshold, is_smaller_then=True)
    probability_red = linear_probability(orientation_diff_red, lower_threshold, upper_threshold, is_smaller_then=True)

    # Combine probabilities using OR logic (union)
    total_probability = probability_union(probability_cyan, probability_red)

    return total_probability


def CloseOrOrientedSameAsRedBoxFn_trial_44(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the blue box is placed close to or oriented the same as the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    next_blue_box_pose = get_pose(next_state, blue_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Calculate the distance between the blue box and the red box
    distance_metric = position_norm_metric(next_blue_box_pose, red_box_pose, norm="L2")
    # Calculate the orientation difference between the blue box and the red box
    orientation_metric = great_circle_distance_metric(next_blue_box_pose, red_box_pose)
    # Define thresholds
    distance_upper_threshold = 0.1  # 10cm for close
    orientation_lower_threshold = torch.pi / 8.0
    orientation_upper_threshold = torch.pi / 6.0
    # Calculate probabilities
    distance_probability = linear_probability(distance_metric, 0.0, distance_upper_threshold, is_smaller_then=True)
    orientation_probability = linear_probability(
        orientation_metric, orientation_lower_threshold, orientation_upper_threshold, is_smaller_then=True
    )
    # Combine probabilities using OR logic
    total_probability = probability_union(distance_probability, orientation_probability)
    return total_probability


CUSTOM_FNS = {
    "CloseOrOrientedSameAsRedBoxFn_trial_44": CloseOrOrientedSameAsRedBoxFn_trial_44,
    "OrientSameAsCyanOrRedBoxFn_trial_43": OrientSameAsCyanOrRedBoxFn_trial_43,
    "OrientLikeCyanBoxFn_trial_42": OrientLikeCyanBoxFn_trial_42,
    "PlaceInLineRightOrBehindBlueBoxFn_trial_41": PlaceInLineRightOrBehindBlueBoxFn_trial_41,
    "PlaceCloseToRedOrBlueBoxFn_trial_40": PlaceCloseToRedOrBlueBoxFn_trial_40,
    "StraightInFrontOfBlueBoxFn_trial_39": StraightInFrontOfBlueBoxFn_trial_39,
    "StraightLineLeftOfRedBoxFn_trial_38": StraightLineLeftOfRedBoxFn_trial_38,
    "StraightLineBlueToFrontFn_trial_37": StraightLineBlueToFrontFn_trial_37,
    "StraightLineCyanToFrontFn_trial_37": StraightLineCyanToFrontFn_trial_37,
    "OrientedSameAsCyanBoxFn_trial_36": OrientedSameAsCyanBoxFn_trial_36,
    "PlaceCloseWithoutTouchingFn_trial_35": PlaceCloseWithoutTouchingFn_trial_35,
    "PlaceInCircleAroundScrewdriver15cmFn_trial_34": PlaceInCircleAroundScrewdriver15cmFn_trial_34,
    "PlaceLeftOfRedBoxFn_trial_33": PlaceLeftOfRedBoxFn_trial_33,
    "TriangleFormationFn_trial_32": TriangleFormationFn_trial_32,
    "FarFromRedAndBlueBoxFn_trial_31": FarFromRedAndBlueBoxFn_trial_31,
    "CloseToBlueBoxFn_trial_31": CloseToBlueBoxFn_trial_31,
    "AlignInLineOnTableFn_trial_30": AlignInLineOnTableFn_trial_30,
    "CloseOrOrientedSameAsRedBoxFn_trial_29": CloseOrOrientedSameAsRedBoxFn_trial_29,
    "OrientationSameAsCyanOrRedBoxFn_trial_28": OrientationSameAsCyanOrRedBoxFn_trial_28,
    "OrientedSameOrOrthogonalToCyanBoxFn_trial_27": OrientedSameOrOrthogonalToCyanBoxFn_trial_27,
    "PlaceRightOrBehindBlueBoxFn_trial_26": PlaceRightOrBehindBlueBoxFn_trial_26,
    "CloseToRedOrBlueBoxFn_trial_25": CloseToRedOrBlueBoxFn_trial_25,
    "StraightInFrontOfBlueBoxFn_trial_24": StraightInFrontOfBlueBoxFn_trial_24,
    "StraightLineLeftOfRedBoxFn_trial_23": StraightLineLeftOfRedBoxFn_trial_23,
    "RedInFrontOfBlueFn_trial_22": RedInFrontOfBlueFn_trial_22,
    "BlueInFrontOfCyanFn_trial_22": BlueInFrontOfCyanFn_trial_22,
    "OrientedSameAsCyanBoxFn_trial_21": OrientedSameAsCyanBoxFn_trial_21,
    "PlaceCloseToOthersFn_trial_20": PlaceCloseToOthersFn_trial_20,
    "CircleAroundScrewdriverFn_trial_19": CircleAroundScrewdriverFn_trial_19,
    "LeftOfRedBoxFn_trial_18": LeftOfRedBoxFn_trial_18,
    "TrianglePlacementCyanBoxFn_trial_17": TrianglePlacementCyanBoxFn_trial_17,
    "TrianglePlacementBlueBoxFn_trial_17": TrianglePlacementBlueBoxFn_trial_17,
    "FarFromRedAndBlueBoxFn_trial_16": FarFromRedAndBlueBoxFn_trial_16,
    "CloseToBlueBoxFn_trial_16": CloseToBlueBoxFn_trial_16,
    "AlignInLineWithBoxesFn_trial_15": AlignInLineWithBoxesFn_trial_15,
    "CloseOrOrientedSameAsRedBoxFn_trial_14": CloseOrOrientedSameAsRedBoxFn_trial_14,
    "OrientLikeCyanOrRedBoxFn_trial_13": OrientLikeCyanOrRedBoxFn_trial_13,
    "OrientLikeOrOrthogonalToCyanBoxFn_trial_12": OrientLikeOrOrthogonalToCyanBoxFn_trial_12,
    "PlaceInLineRightOrBehindBlueBoxFn_trial_11": PlaceInLineRightOrBehindBlueBoxFn_trial_11,
    "CloseToRedOrBlueBoxFn_trial_10": CloseToRedOrBlueBoxFn_trial_10,
    "StraightInFrontOfBlueBoxFn_trial_9": StraightInFrontOfBlueBoxFn_trial_9,
    "StraightLineLeftOfRedBoxFn_trial_8": StraightLineLeftOfRedBoxFn_trial_8,
    "PlaceRedBoxInLineFn_trial_7": PlaceRedBoxInLineFn_trial_7,
    "PlaceBlueBoxInLineFn_trial_7": PlaceBlueBoxInLineFn_trial_7,
    "OrientedSameAsCyanBoxFn_trial_6": OrientedSameAsCyanBoxFn_trial_6,
    "PlaceCloseButNotTouchingFn_trial_5": PlaceCloseButNotTouchingFn_trial_5,
    "CircleAroundScrewdriverFn_trial_4": CircleAroundScrewdriverFn_trial_4,
    "LeftOfRedBoxFn_trial_3": LeftOfRedBoxFn_trial_3,
    "TriangleFormationFn_trial_2": TriangleFormationFn_trial_2,
    "FarFromRedAndBlueBoxFn_trial_1": FarFromRedAndBlueBoxFn_trial_1,
    "CloseToBlueBoxFn_trial_1": CloseToBlueBoxFn_trial_1,
    "InLineWithRedAndBlueBoxFn_trial_0": InLineWithRedAndBlueBoxFn_trial_0,
    "HookHandoverOrientationFn": HookHandoverOrientationFn,
    "ScrewdriverPickFn": ScrewdriverPickFn,
    "ScrewdriverPickActionFn": ScrewdriverPickActionFn,
    "HandoverPositionFn": HandoverPositionFn,
    "HandoverOrientationFn": HandoverOrientationFn,
    "HandoverOrientationAndPositionnFn": HandoverOrientationAndPositionnFn,
    "HandoverVerticalOrientationFn": HandoverVerticalOrientationFn,
    "StraightLeftOfRedBoxFn": StraightLeftOfRedBoxFn,
    "PlaceLeftOfAndNextToRedBoxFn": PlaceLeftOfAndNextToRedBoxFn,
    "StraightInFrontOfBlueBoxFn": StraightInFrontOfBlueBoxFn,
    "StraightInFrontOfCyanBoxFn": StraightInFrontOfCyanBoxFn,
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
    "PlaceNextToRedOrBlueBoxFn": PlaceNextToRedOrBlueBoxFn,
    "StraightRightOfOrBehindBlueBoxFn": StraightRightOfOrBehindBlueBoxFn,
    "OrientedSameOrOrthogonalToCyanBoxFn": OrientedSameOrOrthogonalToCyanBoxFn,
    "OrientedSameAsCyanOrRedBoxFn": OrientedSameAsCyanOrRedBoxFn,
    "PlaceNextToOrOrientedSameAsRedBoxFn": PlaceNextToOrOrientedSameAsRedBoxFn,
}
