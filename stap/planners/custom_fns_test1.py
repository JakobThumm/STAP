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
    distance_to_left_most = position_diff_along_direction(
        next_object_pose, generate_pose_batch([left_most_position, 0, 0, 1, 0, 0, 0], next_object_pose), [1, 0, 0]
    )
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
    alignment_probability = linear_probability(distance_to_line, lower_threshold, upper_threshold, is_smaller_then=True)

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
    probability = linear_probability(combined_distance, lower_threshold, upper_threshold, is_smaller_then=False)
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
    distance_probability_red = linear_probability(
        distance_metric_red, lower_threshold, upper_threshold, is_smaller_then=True
    )
    distance_probability_blue = linear_probability(
        distance_metric_blue, lower_threshold, upper_threshold, is_smaller_then=True
    )
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
    position_diff_cyan_along_direction = position_diff_along_direction(
        cyan_box_next_pose, red_box_pose, direction_vector_red_to_blue
    )
    # Calculate the positional difference of the cyan box normal to the direction vector
    position_diff_cyan_normal_to_direction = position_metric_normal_to_direction(
        cyan_box_next_pose, red_box_pose, direction_vector_red_to_blue
    )

    # The cyan box should be along the line formed by the red and blue boxes, so the normal difference should be minimal
    normal_diff_threshold = 0.05  # Allow a small margin for alignment
    normal_diff_probability = threshold_probability(
        position_diff_cyan_normal_to_direction, normal_diff_threshold, is_smaller_then=True
    )

    # The cyan box should be between the red and blue boxes along the direction vector, so the positional difference should be positive but less than the distance between red and blue
    distance_red_to_blue = position_norm_metric(red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    between_boxes_probability = (position_diff_cyan_along_direction >= 0) & (
        position_diff_cyan_along_direction <= distance_red_to_blue
    )

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
    close_by_probability_blue = linear_probability(
        distance_metric_blue, lower_threshold_blue, upper_threshold_blue, is_smaller_then=True
    )
    # Evaluate if the red box is placed close to the cyan box.
    distance_metric_cyan = position_norm_metric(next_red_box_pose, cyan_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold_cyan = 0.05
    upper_threshold_cyan = 0.10
    close_by_probability_cyan = linear_probability(
        distance_metric_cyan, lower_threshold_cyan, upper_threshold_cyan, is_smaller_then=True
    )
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
