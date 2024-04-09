"""Defines custom evaluation functions to be used in the trajectory evaluation of the planner.

Author: Jakob Thumm
Date: 2024-01-02
"""

from typing import Optional, Sequence, Tuple

import torch

from stap.envs.base import Env, Primitive
from stap.envs.pybullet.table import object_state
from stap.envs.pybullet.table_env import TableEnv
from stap.utils.transformation_utils import (
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_axis_angle,
    rotation_6d_to_matrix,
)


################### UTILITY FUNCTIONS ###################
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
    r"""Returns the orientation of the object as a Rotation matrix.

    Args:
        observation [batch_size, state_dim]: Current state.
        id: ID of the object.

    Returns:
        Orientation of the object [batch_size, 3, 3].
    """
    idxR11 = list(object_state.ObjectState.RANGES.keys()).index("R11")
    idxR21 = list(object_state.ObjectState.RANGES.keys()).index("R21")
    idxR31 = list(object_state.ObjectState.RANGES.keys()).index("R31")
    idxR12 = list(object_state.ObjectState.RANGES.keys()).index("R12")
    idxR22 = list(object_state.ObjectState.RANGES.keys()).index("R22")
    idxR32 = list(object_state.ObjectState.RANGES.keys()).index("R32")
    rotations = rotation_6d_to_matrix(observation[:, id : id + 1, [idxR11, idxR21, idxR31, idxR12, idxR22, idxR32]])
    return rotations[:, 0, :, :]


############## HELPER FUNCTIONS FOR THE CUSTOM FNS ##############
def get_object_id_from_name(name: str, env: Env) -> int:
    """Return the object identifier from a given object name."""
    assert isinstance(env, TableEnv)
    return env.get_object_id_from_name(name)


def get_object_id_from_primitive(arg_id: int, primitive: Primitive) -> int:
    """Return the object identifier from a primitive and its argument id.

    Example: The primitive `Place` has two argument ids: `object` with `arg_id = 0` and `target` with `arg_id = 1`.
    """
    arg_object_ids = primitive.get_policy_args_ids()
    return arg_object_ids[arg_id]


def get_pose(state: torch.Tensor, object_id: int, frame: int = -1) -> torch.Tensor:
    """Return the pose of an object in the requested frame.

    Args:
        state: state (observation) to extract the pose from.
        object_id: number identifying the obect. Can be retrieved with `get_object_id_from_name()` and
            `get_object_id_from_primitive()`.
        frame: the frame to represent the pose in. Default is `-1`, which is world frame. In our simulation, the base
            frame equals the world frame. Give the object id for other frames, e.g., `0` for end effector frame.
    Returns:
        The pose in shape [..., 7] with format [x, y, z, qw, qx, qy, qz], with the rotation represented as a quaternion.
    """
    object_pose = torch.zeros([state.shape[0], 7], device=state.device)
    object_pose[:, :3] = get_object_position(state, object_id)
    object_rotation_matrix = get_object_orientation(state, object_id)
    if frame == -1:
        object_pose[:, 3:] = matrix_to_quaternion(object_rotation_matrix)
    else:
        assert frame >= 0, "Unknown frame."
        frame_position = get_object_position(state, frame)
        frame_rotation_matrix = get_object_orientation(state, frame)
        # Compute the inverse of the object's rotation matrix
        R_frame_inv = frame_rotation_matrix.transpose(-2, -1)
        # Calculate the relative position
        relative_position = torch.matmul(R_frame_inv, (object_pose[:, :3] - frame_position).unsqueeze(-1)).squeeze(-1)
        object_pose[:, :3] = relative_position
        # Calculate relative orientation
        relative_rotation_matrix = torch.matmul(R_frame_inv, object_rotation_matrix)
        object_pose[:, 3:] = matrix_to_quaternion(relative_rotation_matrix)
        # Calculate the relative position in object frame

    return object_pose


def position_norm_metric(
    pose_1: torch.Tensor, pose_2: torch.Tensor, norm: str = "L2", axes: Sequence[str] = ["x", "y", "z"]
) -> torch.Tensor:
    """Calculate the norm of the positional difference of two poses along the given axes.

    Args:
        pose_{1, 2}: the poses of the two objects.
        norm: which norm to calculate. Choose from 'L1', 'L2', and 'Linf'. Defaults to `L2`.
        axes: calculate the norm along the given axes and ignore all other axes. Choose entries from `{'x', 'y', 'z'}`.
    Returns:
        The norm in shape [..., 1]
    """
    assert norm in ["L1", "L2", "Linf"], "Unknown norm."
    assert all([axis in ["x", "y", "z"] for axis in axes]), "Unknown axis."
    assert len(axes) > 0, "No axes given."
    axes_binary = [1 if axis in axes else 0 for axis in ["x", "y", "z"]]
    position_diff = pose_1[..., :3] - pose_2[..., :3]
    position_diff = position_diff * torch.tensor(axes_binary, device=position_diff.device)
    if norm == "L1":
        return torch.norm(position_diff, p=1, dim=-1, keepdim=True)
    elif norm == "L2":
        return torch.norm(position_diff, p=2, dim=-1, keepdim=True)
    elif norm == "Linf":
        return torch.norm(position_diff, p=float("inf"), dim=-1, keepdim=True)
    else:
        raise NotImplementedError()


def great_circle_distance_metric(pose_1: torch.Tensor, pose_2: torch.Tensor) -> torch.Tensor:
    """Calculate the difference in orientation in radians of two poses using the great circle distance.

    Assumes that the position entries of the poses are direction vectors `v1` and `v2`.
    The great circle distance is then `d = arccos(dot(v1, v2))` in radians.
    """
    eps = 1e-6
    v1 = pose_1[..., :3] / torch.norm(pose_1[..., :3], dim=-1, keepdim=True)
    v2 = pose_2[..., :3] / torch.norm(pose_2[..., :3], dim=-1, keepdim=True)
    return torch.acos(torch.clip(torch.sum(v1 * v2, dim=-1, keepdim=True), -1.0 + eps, 1.0 - eps))


def pointing_in_direction_metric(
    pose_1: torch.Tensor, pose_2: torch.Tensor, main_axis: Sequence[float] = [1, 0, 0]
) -> torch.Tensor:
    """Evaluate if an object is pointing in a given direction.

    Rotates the given main axis by the rotation of pose_1 and calculates the `great_circle_distance()`
    between the rotated axis and pose_2.position.
    Args:
        pose_1: the orientation of this pose is used to rotate the `main_axis`.
        pose_2: compare the rotated `main_axis` with the position vector of this pose.
        main_axis: axis describing in which direction an object is pointing in its default configuration.
    Returns:
        The great circle distance in radians between the rotated `main_axis` and the position part of `pose_2`.
    """
    main_axis = torch.FloatTensor(main_axis).to(pose_1.device)  # type: ignore
    norm_main_axis = main_axis / torch.norm(main_axis, dim=-1, keepdim=True)
    new_pose = torch.zeros_like(pose_1)
    new_pose[..., :3] = quaternion_apply(pose_1[..., 3:], norm_main_axis)
    return great_circle_distance_metric(new_pose, pose_2)


def rotation_angle_metric(pose_1: torch.Tensor, pose_2: torch.Tensor, axis: Sequence[float]) -> torch.Tensor:
    """Calculate the rotational difference between pose_1 and pose_2 around the given axis.

    Example: The orientation 1 is not rotated and the orientation 2 is rotated around the z-axis by 90 degree.
        Then if the given axis is [0, 0, 1], the function returns pi/2.
        If the given axis is [1, 0, 0], the function returns 0, as there is no rotation around the x-axis.

    Args:
        pose_{1, 2}: the orientations of the two poses are used to calculate the rotation angle.
        axis: The axis of interest to rotate around.

    Returns:
        The angle difference in radians along the given axis.
    """
    relative_rotation = quaternion_multiply(quaternion_invert(pose_1[..., 3:]), pose_2[..., 3:])
    relative_rotation_rot_vec = quaternion_to_axis_angle(relative_rotation)
    rotation_axis = torch.zeros_like(relative_rotation_rot_vec)
    rotation_axis[..., 0] = axis[0]
    rotation_axis[..., 1] = axis[1]
    rotation_axis[..., 2] = axis[2]
    # dot product of the rotation axis and the relative rotation axis
    dot_product = torch.sum(rotation_axis * relative_rotation_rot_vec, dim=-1, keepdim=True)
    return dot_product


def threshold_probability(metric: torch.Tensor, threshold: float, is_smaller_then: bool = True) -> torch.Tensor:
    """If `is_smaller_then`: return `1.0` if `metric < threshold` and `0.0` otherwise.
    If not `is_smaller_then`: return `1.0` if `metric >= threshold` and `0.0` otherwise.
    """
    cdf = torch.where(metric < threshold, 1.0, 0.0)[:, 0]
    if not is_smaller_then:
        cdf = 1.0 - cdf
    return cdf


def linear_probability(
    metric: torch.Tensor, lower_threshold: float, upper_threshold: float, is_smaller_then: bool = True
) -> torch.Tensor:
    """Return the linear probility given a metric and two thresholds.

    If `is_smaller_then` return:
        - `1.0` if `metric < lower_threshold`
        - `0.0` if `metric < upper_threshold`
        - linearly interpolate between 0 and 1 otherwise.
    If not `is_smaller_then` return:
        - `1.0` if `metric >= upper_threshold`
        - `0.0` if `metric < lower_threshold`
        - linearly interpolate between 1 and 0 otherwise.
    """
    cdf = torch.clip((metric - lower_threshold) / (upper_threshold - lower_threshold), 0.0, 1.0)[:, 0]
    if is_smaller_then:
        cdf = 1.0 - cdf
    return cdf


def normal_probability(metric: torch.Tensor, mean: float, std_dev: float, is_smaller_then: bool = True) -> torch.Tensor:
    """Return a probability function based on the cummulative distribution function with `mean` and `std_dev`.

    Args:
        metric: the metric to calculate the value for.
        mean: the mean of the cummulative distribution function.
        std_dev: the standard deviation of the cummulative distribution function.
        is_smaller_then: if true, invert the return value with `(1-p)`.
    Returns:
        `cdf(metric, mean, std_dev)`
    """
    cdf = 0.5 * (1 + torch.erf((metric - mean) / (std_dev * torch.sqrt(torch.tensor(2.0, device=metric.device)))))[:, 0]
    if is_smaller_then:
        cdf = 1.0 - cdf
    return cdf


def probability_intersection(p_1: torch.Tensor, p_2: torch.Tensor) -> torch.Tensor:
    """Calculate the intersection of two probabilities `p = p_1 * p_2`."""
    return p_1 * p_2


def probability_union(p_1: torch.Tensor, p_2: torch.Tensor) -> torch.Tensor:
    """Calculate the union of two probabilities `p = max(p_1, p_2)`."""
    return torch.max(p_1, p_2)


################ OLD FUNCTIONS #################
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
    R_object = get_object_orientation(observation, obj_id)
    eef_position = get_object_position(observation, eef_id)
    R_eef = get_object_orientation(observation, eef_id)

    # Compute the inverse of the object's rotation matrix
    R_object_inv = R_object.transpose(-2, -1)

    # Calculate the relative position
    relative_position = torch.matmul(R_object_inv, (eef_position - object_position).unsqueeze(-1)).squeeze(-1)

    # Calculate relative orientation
    R_relative = torch.matmul(R_object_inv, R_eef)

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
    end_effector_id = get_object_id_from_name("end_effector", env)
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
    threshold_greater = 0.0
    position_value_1 = MAX_VALUE * (action[:, 0] > threshold_greater) + MIN_VALUE * (action[:, 0] < threshold_greater)
    threshold_smaller = 0.3
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


def StaticHandoverPreferenceFnChris(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    hand_id = get_object_id_from_primitive(1, primitive)
    next_object_pose = get_pose(next_state, object_id)
    current_hand_pose = get_pose(state, hand_id)
    next_hand_pose = get_pose(next_state, hand_id)

    orientation_metric = pointing_in_direction_metric(next_object_pose, current_hand_pose, main_axis=[-1, 0, 0])
    position_metric = position_norm_metric(next_object_pose, next_hand_pose, norm="L2", axes=["x", "y", "z"])

    # Considering the human preference to have the handover closer
    lower_threshold_orientation = torch.pi / 6.0
    upper_threshold_orientation = torch.pi / 4.0
    lower_threshold_distance = 0.4  # Preference for closer handover than 0.2m
    upper_threshold_distance = 0.8  # Upper bound reduced from an arm's length

    probability_orientation = linear_probability(
        orientation_metric, lower_threshold_orientation, upper_threshold_orientation, is_smaller_then=True
    )
    probability_distance = linear_probability(
        position_metric, lower_threshold_distance, upper_threshold_distance, is_smaller_then=True
    )

    total_probability = probability_intersection(probability_orientation, probability_distance)
    return total_probability


def HandoverParallelAndOrientationFnCaroline(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the orientation of the screwdriver handover ensuring it is also parallel to the table.

    Please make sure that the screwdriver is parallel to the table when handing it over. It was tilted a few times.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: Optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    hand_id = get_object_id_from_primitive(1, primitive)
    next_object_pose = get_pose(next_state, object_id)
    current_hand_pose = get_pose(state, hand_id)

    handle_main_axis = [-1.0, 0.0, 0.0]  # Main axis for orientation towards the hand
    table_normal = [0.0, 0.0, 1.0]  # Normal vector for the table (z-axis in world frame is upwards)

    # Orientation towards the hand
    orientation_metric = pointing_in_direction_metric(next_object_pose, current_hand_pose, handle_main_axis)
    orientation_prob = linear_probability(orientation_metric, torch.pi / 6.0, torch.pi / 4.0, is_smaller_then=True)

    # Checking if parallel to the table by ensuring the main axis of the screwdriver (z-axis) is aligned with the table's normal
    parallel_metric = pointing_in_direction_metric(
        next_object_pose, torch.tensor(table_normal).to(next_state.device), [0.0, 0.0, 1.0]
    )
    parallel_prob = threshold_probability(parallel_metric, torch.pi / 18.0)  # ~10 degrees of allowance

    # Combining both probabilities
    total_probability = probability_intersection(orientation_prob, parallel_prob)

    return total_probability


def StaticHandoverCloseProximityFnBenedikt(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    """
    Evaluates the orientation and position of the screwdriver during a static handover to the right hand, focusing on
    close proximity to the human partner as per request.
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    hand_id = get_object_id_from_name("right_hand", env)
    next_object_pose = get_pose(next_state, object_id)
    next_hand_pose = get_pose(next_state, hand_id)

    # Orienting the handle to be easily graspable by the human hand
    handle_main_axis = [-1.0, 0.0, 0.0]  # Considering default handle orientation
    orientation_metric = pointing_in_direction_metric(next_object_pose, next_hand_pose, handle_main_axis)
    orientation_prob = linear_probability(orientation_metric, torch.pi / 6.0, torch.pi / 4.0, is_smaller_then=True)

    # Ensuring closer proximity during handover
    position_metric = position_norm_metric(next_object_pose, next_hand_pose, norm="L2", axes=["x", "y", "z"])
    # Adjusting for closer proximity preference by the human partner
    closer_lower_threshold = 0.2  # Desired closer starting point of proximity
    closer_upper_threshold = 0.5  # Reducing the acceptable maximum distance for better comfort
    proximity_prob = linear_probability(
        position_metric, closer_lower_threshold, closer_upper_threshold, is_smaller_then=True
    )

    # Combining both orientation and proximity preferences
    total_preference = probability_intersection(proximity_prob, orientation_prob)
    return total_preference


def PickScrewdriverPreferenceFnRobin(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    """
    Prefer picking the screwdriver by the rod, so its handle points away from the robot,
    making it safer and more natural for the handover phase.
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    end_effector_id = get_object_id_from_name("end_effector", env)

    next_end_effector_pose = get_pose(next_state, end_effector_id, object_id)

    preferred_grasp_pose = torch.FloatTensor([0.075 / 2.0, 0, 0, 1, 0, 0, 0]).to(next_state.device)

    position_metric = position_norm_metric(next_end_effector_pose, preferred_grasp_pose, norm="L2", axes=["x"])

    probability_grasp = threshold_probability(position_metric, 0.075 / 2.0, is_smaller_then=True)

    return probability_grasp


def PickScrewdriverPreferenceFnRobin2(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    """
    Adjusts the preference for picking the screwdriver to ensure a stable grip.
    Prefers picking the screwdriver by the central part of the rod, away from the very tip,
    to ensure stability while still allowing for a comfortable handover.
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    end_effector_id = get_object_id_from_name("end_effector", env)

    next_end_effector_pose = get_pose(next_state, end_effector_id, object_id)

    # Adjust preference to grasp closer to the central part of the rod for stability.
    # Assume the rod's suitable grasp range starts 1/4th from the tip towards the handle.
    rod_length = 0.075  # Total length of the rod
    preferred_grasp_start = rod_length * 1 / 4  # Start of the preferred grasp region from the tip
    preferred_grasp_end = rod_length * 3 / 4  # End of the preferred grasp region, closer to the handle

    # Calculate the positional norm metric for the grasp position along the rod's length (x-axis in object frame).
    position_metric = position_norm_metric(
        next_end_effector_pose,
        torch.FloatTensor([rod_length / 2.0, 0, 0, 1, 0, 0, 0]).to(next_state.device),
        norm="L2",
        axes=["x"],
    )

    # Use a linear probability distribution within the preferred grasp range.
    probability_grasp_stability = linear_probability(
        position_metric, preferred_grasp_start, preferred_grasp_end, is_smaller_then=False
    )

    return probability_grasp_stability


def StaticHandoverScrewdriverPreferenceFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    """
    Evaluates the orientation and position of the screwdriver during the handover to ensure the handle points towards
    the human's right hand, and the rod points away for safety reasons.
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    screwdriver_id = get_object_id_from_primitive(0, primitive)
    right_hand_id = get_object_id_from_primitive(1, primitive)

    next_screwdriver_pose = get_pose(next_state, screwdriver_id)
    current_right_hand_pose = get_pose(state, right_hand_id)

    handle_main_axis = [-1.0, 0.0, 0.0]  # Assuming the handle points in the negative x direction in its frame

    orientation_metric = pointing_in_direction_metric(next_screwdriver_pose, current_right_hand_pose, handle_main_axis)

    # Setting conservative thresholds to ensure safety and natural handover motion
    lower_threshold = torch.pi / 12.0  # Tighter threshold for more precision
    upper_threshold = torch.pi / 6.0

    probability_orientation = linear_probability(
        orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )

    # Ensuring distance to right hand is within a comfortable range for handover
    position_metric = position_norm_metric(
        next_screwdriver_pose, current_right_hand_pose, norm="L2", axes=["x", "y", "z"]
    )
    distance_lower_threshold = 0.2  # Closer proximity is preferred
    distance_upper_threshold = 0.5  # Beyond this distance, the orientation becomes less relevant

    probability_position = linear_probability(
        position_metric, distance_lower_threshold, distance_upper_threshold, is_smaller_then=True
    )

    # Using intersection to ensure both orientation and position are considered together for safety
    total_probability = probability_intersection(probability_position, probability_orientation)

    return total_probability


def ScrewdriverHandoverFnPriya(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the screwdriver's handle is pointing upwards during the handover.
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
    handle_main_axis = [-1.0, 0.0, 0.0]  # The handle points in negative x direction in the object frame

    # We want to know if the handle is pointing upwards after the handover action.
    # Create a direction vector pointing upwards in the world frame
    direction_upwards = torch.tensor([0.0, 0.0, 1.0], device=next_state.device).expand_as(next_object_pose[:, :3])

    # Calculate the orientation metric for the handle pointing upwards
    orientation_metric_up = pointing_in_direction_metric(next_object_pose, direction_upwards, handle_main_axis)

    # Define thresholds for considering the handle to be sufficiently pointing upwards
    lower_threshold = torch.pi / 4.0  # 30 degrees
    upper_threshold = torch.pi / 3.0  # 45 degrees

    # Calculate the probabilit  y that the handle is pointing upwards within the acceptable range
    probability_handover_up = linear_probability(
        orientation_metric_up, lower_threshold, upper_threshold, is_smaller_then=True
    )

    return probability_handover_up


def ScrewdriverHandoverFnBrandon(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the orientation and position of the screwdriver during handover to ensure the handle is handed over.
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
    handle_main_axis = [-1.0, 0.0, 0.0]  # Assuming the handle points in negative x direction

    # Ensure the handle is oriented towards the human's hand
    orientation_metric = pointing_in_direction_metric(next_object_pose, current_hand_pose, handle_main_axis)
    lower_threshold_orientation = torch.pi / 6.0  # 30 degrees
    upper_threshold_orientation = torch.pi / 4.0  # 45 degrees
    probability_handover_orientation = linear_probability(
        orientation_metric, lower_threshold_orientation, upper_threshold_orientation, is_smaller_then=True
    )

    # Ensure the screwdriver is within a reachable distance to the human's hand
    position_metric = position_norm_metric(next_object_pose, next_hand_pose, norm="L2", axes=["x", "y", "z"])
    lower_threshold_position = 0.2  # Preferred close distance in meters
    upper_threshold_position = 0.8  # Maximum acceptable distance in meters
    probability_handover_position = linear_probability(
        position_metric, lower_threshold_position, upper_threshold_position, is_smaller_then=True
    )

    # Combine the orientation and position probabilities
    total_probability = probability_intersection(probability_handover_orientation, probability_handover_position)
    return total_probability


def ScrewdriverHandoverVerticalFnRobert(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the screwdriver is presented vertically during the handover.
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
    # We want to know if the handle is pointing vertically after the handover action.
    # This is evaluated by checking if the handle's main axis is aligned with the global z-axis.
    vertical_direction = torch.zeros_like(next_object_pose)
    vertical_direction[:, 2] = 1.0  # Global z-axis
    orientation_metric_vertical = pointing_in_direction_metric(next_object_pose, vertical_direction, handle_main_axis)
    # Thresholds for considering the screwdriver to be vertical
    lower_threshold = torch.pi / 3.0  # 30 degrees
    upper_threshold = torch.pi / 2.0  # 45 degrees
    # Calculate the probability
    probability_handover_vertical = linear_probability(
        orientation_metric_vertical, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return probability_handover_vertical


def ScrewdriverHandoverFnRobert2(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the screwdriver tip is never pointing at the human during handover.
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
    human_id = get_object_id_from_primitive(1, primitive)
    next_object_pose = get_pose(next_state, object_id)
    human_pose = get_pose(state, human_id)
    rod_main_axis = [1.0, 0.0, 0.0]  # The rod points in the positive x direction of the object frame
    # We want to ensure the rod (tip) is never pointing at the human during handover.
    orientation_metric = pointing_in_direction_metric(next_object_pose, human_pose, rod_main_axis)
    # A threshold to consider the screwdriver as not pointing directly at the human.
    # This threshold is somewhat arbitrary and could be adjusted based on safety requirements.
    threshold = torch.pi / 4.0  # 45 degrees
    # Calculate the probability
    probability_not_pointing_at_human = threshold_probability(orientation_metric, threshold, is_smaller_then=True)
    return probability_not_pointing_at_human


def ScrewdriverHandoverFnJohn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the screwdriver handle is pointing upwards during the handover.
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
    # We want to know if the handle is pointing upwards after the handover action.
    direction = torch.zeros_like(next_object_pose)
    direction[:, 2] = 1.0  # Upwards direction in z-axis
    orientation_metric_up = pointing_in_direction_metric(next_object_pose, direction, handle_main_axis)
    lower_threshold = torch.pi / 6.0  # 30 degrees
    upper_threshold = torch.pi / 3.0  # 45 degrees
    # Calculate the probability
    probability_handover_up = linear_probability(
        orientation_metric_up, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return probability_handover_up


def ScrewdriverHandoverCloseFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the closeness of the screwdriver handover to the human's right hand.
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

    # We want the handover to be as close as possible to the human's hand.
    position_metric = position_norm_metric(next_object_pose, current_hand_pose, norm="L2", axes=["x", "y", "z"])

    # Define thresholds for closeness. Closer than 0.2m is preferred, while further than 0.5m is less preferred.
    lower_threshold = 0.1
    upper_threshold = 0.8

    # Calculate the probability based on the closeness metric.
    probability_handover_closeness = linear_probability(
        position_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )

    return probability_handover_closeness


def ScrewdriverHandoverFnSarah(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the handover of the screwdriver, ensuring it is close and the handle points downwards.
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
    # We want to know if the handle is pointing downwards after the handover action.
    direction = torch.zeros_like(next_object_pose)
    direction[:, 2] = -1.0
    orientation_metric_down = pointing_in_direction_metric(next_object_pose, direction, handle_main_axis)
    lower_threshold = torch.pi / 6.0
    upper_threshold = torch.pi / 3.0
    # Calculate the probability for orientation
    probability_handover_orientation = linear_probability(
        orientation_metric_down, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # We want to be close to the human hand.
    position_metric = position_norm_metric(next_object_pose, current_hand_pose, norm="L2", axes=["x", "y", "z"])
    # Handing over the object an arm length ~0.8m away is considered a failure and close ~0.2m is preferred.
    lower_threshold = 0.2
    upper_threshold = 0.8
    probability_handover_position = linear_probability(
        position_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine the probabilities for final evaluation
    total_probability = probability_intersection(probability_handover_position, probability_handover_orientation)
    return total_probability


def ScrewdriverHandoverFnJakob1(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the orientation and position of the screwdriver handover to ensure the handle points towards the human and is closer.
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
    lower_threshold_orientation = torch.pi / 6.0
    upper_threshold_orientation = torch.pi / 3.0
    # Calculate the probability for orientation
    probability_handover_orientation = linear_probability(
        orientation_metric, lower_threshold_orientation, upper_threshold_orientation, is_smaller_then=True
    )
    # We want the handover to be closer to the human hand.
    position_metric = position_norm_metric(next_object_pose, current_hand_pose, norm="L2", axes=["x", "y", "z"])
    mean = 0.0  # Closer to the hand is preferred
    std = 0.1  # standard deviation
    # Calculate the probability for position
    probability_handover_position = normal_probability(position_metric, mean, std, is_smaller_then=True)
    # Combine the probabilities for orientation and position
    total_probability = probability_intersection(probability_handover_orientation, probability_handover_position)
    return total_probability


def ScrewdriverHandoverFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the screwdriver handle is pointing upwards and towards the human in a 45-degree angle during handover.
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

    # Calculate if the handle is pointing upwards and towards the human in a 45-degree angle.
    # Assuming the human is facing the robot, the direction towards the human would be along the negative y-axis.
    direction_towards_human = torch.tensor([0.0, -1.0, 1.0]).to(next_state.device).normalize()
    orientation_metric = pointing_in_direction_metric(next_object_pose, direction_towards_human, handle_main_axis)

    # A 45-degree angle in radians.
    target_angle = torch.pi / 4.0
    # Allow a small margin around the 45-degree angle for practical reasons.
    lower_threshold = target_angle - torch.pi / 12.0  # 45 degrees - 10 degrees
    upper_threshold = target_angle + torch.pi / 12.0  # 45 degrees + 10 degrees

    # Calculate the probability for orientation.
    probability_orientation = linear_probability(
        orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )

    # We want the screwdriver to be close to the human hand.
    position_metric = position_norm_metric(next_object_pose, current_hand_pose, norm="L2", axes=["x", "y", "z"])
    # Assuming a comfortable handover distance is within 0.2m to 0.4m.
    mean = 0.0  # Closer to the hand is preferred
    std = 0.1  # standard deviation
    # Calculate the probability for position
    probability_handover_position = normal_probability(position_metric, mean, std, is_smaller_then=True)

    # Combine the probabilities for orientation and position.
    total_probability = probability_intersection(probability_orientation, probability_handover_position)
    return total_probability


def ScrewdriverHandoverFnWuang(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the screwdriver is handed over at a higher position.
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
    # Define the desired minimum height for the handover to be considered "higher"
    desired_min_height = 0.45  # This value can be adjusted based on the specific requirements or environment setup

    # Extract the z-coordinate (height) from the next object pose
    z_coordinate = position_norm_metric(next_object_pose, torch.zeros_like(next_object_pose), norm="L2", axes=["z"])

    # Calculate the probability based on whether the z-coordinate meets or exceeds the desired minimum height
    # Here, we use a simple threshold-based approach for demonstration purposes
    probability_handover_height = threshold_probability(z_coordinate, desired_min_height, is_smaller_then=False)

    return probability_handover_height


def ScrewdriverHandoverFnJoey(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the screwdriver handle is pointing upwards during the handover.
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
    # We want to know if the handle is pointing upwards after the handover action.
    direction = torch.zeros_like(next_object_pose)
    direction[:, 2] = 1.0  # Upwards direction in z-axis
    orientation_metric_up = pointing_in_direction_metric(next_object_pose, direction, handle_main_axis)
    lower_threshold = torch.pi / 6.0  # 30 degrees
    upper_threshold = torch.pi / 3.0  # 45 degrees
    # Calculate the probability
    probability_handover_up = linear_probability(
        orientation_metric_up, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return probability_handover_up


def ScrewdriverHandoverFnGiolle(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the handover with a focus on maximizing the distance between the end effector and the handle.
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
    end_effector_id = 0
    next_object_pose = get_pose(next_state, object_id)
    next_end_effector_pose = get_pose(next_state, end_effector_id)

    # Calculate the distance between the end effector and the handle's origin point.
    # Assuming the handle's origin is at the point where the rod and the handle meet.
    distance_metric = position_norm_metric(next_end_effector_pose, next_object_pose, norm="L2", axes=["x", "y", "z"])

    # Since we want to maximize this distance, we can use a threshold to ensure it's sufficiently large.
    # The threshold can be adjusted based on the robot's reach and the user's comfort level.
    threshold_distance = 0.1  # Example threshold, can be adjusted.

    # Calculate the probability that the distance is greater than the threshold.
    probability_distance = threshold_probability(distance_metric, threshold_distance, is_smaller_then=False)

    return probability_distance


def ScrewdriverPickFnDavide(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the preference for picking the screwdriver at the beginning of the handle.
    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed pick [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    end_effector_id = get_object_id_from_name("end_effector", env)
    # Get the pose of the end effector in the object frame
    next_end_effector_pose = get_pose(next_state, end_effector_id, object_id)
    # Assumes the handle length is 0.090 and the handle in the negative x direction in object frame.
    preferred_pick_pose = torch.FloatTensor([-0.090 / 2.0, 0, 0, 1, 0, 0, 0]).to(next_state.device)
    # Calculate the positional norm metric
    position_metric = position_norm_metric(next_end_effector_pose, preferred_pick_pose, norm="L2", axes=["x"])
    # Calculate the probability
    probability_pick_handle = threshold_probability(position_metric, 0.090 / 2.0, is_smaller_then=True)
    return probability_pick_handle


def ScrewdriverHandoverFnDavide(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the screwdriver handle is pointing upwards during handover.
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
    # We want to know if the handle is pointing upwards after the handover action.
    direction = torch.zeros_like(next_object_pose)
    direction[:, 2] = 1.0
    orientation_metric_up = pointing_in_direction_metric(next_object_pose, direction, handle_main_axis)
    lower_threshold = torch.pi / 6.0
    upper_threshold = torch.pi / 4.0
    # Calculate the probability
    probability_handover_up = linear_probability(
        orientation_metric_up, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return probability_handover_up


def ScrewdriverHandoverFnDistanceTableLow(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the screwdriver is facing upwards or downwards during handover and ensures it's a bit lower.
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
    table_id = get_object_id_from_name("table", env)
    table_pose = get_pose(state, table_id)

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
    orientation_probability = probability_union(probability_handover_up, probability_handover_down)

    # Ensure the handover is a bit lower than before, closer to the table surface
    z_distance_to_table = next_object_pose[:, 2] - table_pose[:, 2]
    # Assuming the table height is 0.05 and we want the handover to be just above it
    desired_z_distance = 0.25  # Table height + desired offset from the table
    z_distance_metric = position_norm_metric(
        next_object_pose, torch.zeros_like(next_object_pose), norm="L2", axes=["z"]
    )
    # Smaller distance is better, so we use a threshold probability
    probability_z_distance = threshold_probability(z_distance_metric, desired_z_distance, is_smaller_then=True)

    # Combine the orientation and z-distance probabilities
    total_probability = probability_intersection(orientation_probability, probability_z_distance)
    return total_probability


def ScrewdriverPickFnHugo(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the preference for picking the screwdriver at the beginning of the handle.
    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed pick [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the pose of the screwdriver in the world frame
    screwdriver_pose = get_pose(next_state, object_id)
    # Define the preferred position for picking the screwdriver at the beginning of the handle
    # Assuming the handle length is 0.090, we offset by a small amount in the negative x direction
    preferred_pick_position = torch.FloatTensor([-0.005, 0, 0, 1, 0, 0, 0]).to(next_state.device)
    # Calculate the positional norm metric
    position_metric = position_norm_metric(screwdriver_pose, preferred_pick_position, norm="L2", axes=["x"])
    # Define a threshold for how close the pick action needs to be to the preferred position
    threshold = 0.02  # 1cm tolerance
    # Calculate the probability
    probability_pick_preference = threshold_probability(position_metric, threshold, is_smaller_then=True)
    return probability_pick_preference


def ScrewdriverHandoverFnDaniele(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the screwdriver's handle is pointing upwards during the handover.
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
    handle_main_axis = [-1.0, 0.0, 0.0]  # The handle points in negative x direction in the object frame

    # We want to know if the handle is pointing upwards after the handover action.
    # Create a direction vector pointing upwards in the world frame
    direction_upwards = torch.tensor([0.0, 0.0, 1.0], device=next_state.device).expand_as(next_object_pose[:, :3])

    # Calculate the orientation metric for the handle pointing upwards
    orientation_metric_up = pointing_in_direction_metric(next_object_pose, direction_upwards, handle_main_axis)

    # Define thresholds for considering the handle to be sufficiently pointing upwards
    lower_threshold = torch.pi / 6.0  # 30 degrees
    upper_threshold = torch.pi / 3.0  # 45 degrees

    # Calculate the probability that the handle is pointing upwards within the acceptable range
    probability_handover_up = linear_probability(
        orientation_metric_up, lower_threshold, upper_threshold, is_smaller_then=True
    )

    return probability_handover_up


def ScrewdriverHandoverFnDaniele2(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates if the screwdriver handle is pointing downwards during the handover.
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
    # We want to know if the handle is pointing downwards after the handover action.
    direction = torch.zeros_like(next_object_pose)
    direction[:, 2] = -1.0  # Downwards direction in z-axis
    orientation_metric_down = pointing_in_direction_metric(next_object_pose, direction, handle_main_axis)
    lower_threshold = torch.pi / 6.0  # 30 degrees tolerance
    upper_threshold = torch.pi / 3.0  # 45 degrees tolerance
    # Calculate the probability
    probability_handover_down = linear_probability(
        orientation_metric_down, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return probability_handover_down


def ScrewdriverPickFn11(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the preference for picking the screwdriver further at the tip of the rod.
    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed pick [batch_size] \in [0, 1].
    """
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    end_effector_id = get_object_id_from_name("end_effector", env)
    next_end_effector_pose = get_pose(next_state, end_effector_id, object_id)
    # Assuming the rod length is 0.075 and we want to grasp it further at the tip.
    preferred_grasp_pose = torch.FloatTensor([0.075 * 0.75, 0, 0, 1, 0, 0, 0]).to(next_state.device)
    position_metric = position_norm_metric(next_end_effector_pose, preferred_grasp_pose, norm="L2", axes=["x"])
    # Calculate the probability
    probability_grasp_tip = threshold_probability(position_metric, 0.075 * 0.5, is_smaller_then=True)
    return probability_grasp_tip


def ScrewdriverHandoverFn11(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the preference for handing over the screwdriver as close as possible to the human's right hand.
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
    # We want the screwdriver to be as close as possible to the human hand.
    position_metric = position_norm_metric(next_object_pose, current_hand_pose, norm="L2", axes=["x", "y", "z"])
    # Calculate the probability
    mean = 0.0  # Closer to the hand is preferred
    std = 0.1  # standard deviation
    # Calculate the probability for position
    probability_handover_position = normal_probability(position_metric, mean, std, is_smaller_then=True)
    return probability_handover_position


def ScrewdriverHandoverFn12(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r"""Evaluates the handover of the screwdriver by the handle and proximity to the human hand.
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
    lower_threshold_orientation = torch.pi / 6.0
    upper_threshold_orientation = torch.pi / 4.0
    # Calculate the probability for orientation
    probability_handover_orientation = linear_probability(
        orientation_metric, lower_threshold_orientation, upper_threshold_orientation, is_smaller_then=True
    )
    # We want the handover to be as close to the human hand as possible.
    position_metric = position_norm_metric(next_object_pose, current_hand_pose, norm="L2", axes=["x", "y", "z"])
    # Calculate the probability
    mean = 0.0  # Closer to the hand is preferred
    std = 0.1  # standard deviation
    # Calculate the probability for position
    probability_handover_position = normal_probability(position_metric, mean, std, is_smaller_then=True)
    # Combine the probabilities for orientation and position
    total_probability = probability_intersection(probability_handover_orientation, probability_handover_position)
    return total_probability


CUSTOM_FNS = {
    "HookHandoverOrientationFn": HookHandoverOrientationFn,
    "ScrewdriverPickFn": ScrewdriverPickFn,
    "ScrewdriverPickActionFn": ScrewdriverPickActionFn,
    "HandoverPositionFn": HandoverPositionFn,
    "HandoverOrientationFn": HandoverOrientationFn,
    "HandoverOrientationAndPositionnFn": HandoverOrientationAndPositionnFn,
    "HandoverVerticalOrientationFn": HandoverVerticalOrientationFn,
    "StaticHandoverPreferenceFnChris": StaticHandoverPreferenceFnChris,
    "PickScrewdriverPreferenceFnRobin": PickScrewdriverPreferenceFnRobin,
    "PickScrewdriverPreferenceFnRobin2": PickScrewdriverPreferenceFnRobin2,
    "StaticHandoverScrewdriverPreferenceFn": StaticHandoverScrewdriverPreferenceFn,
    "ScrewdriverHandoverFnPriya": ScrewdriverHandoverFnPriya,
    "ScrewdriverHandoverFnBrandon": ScrewdriverHandoverFnBrandon,
    "ScrewdriverHandoverVerticalFnRobert": ScrewdriverHandoverVerticalFnRobert,
    "ScrewdriverHandoverFnRobert2": ScrewdriverHandoverFnRobert2,
    "ScrewdriverHandoverFnJohn": ScrewdriverHandoverFnJohn,
    "ScrewdriverHandoverCloseFn": ScrewdriverHandoverCloseFn,
    "ScrewdriverHandoverFnSarah": ScrewdriverHandoverFnSarah,
    "ScrewdriverHandoverFnJakob1": ScrewdriverHandoverFnJakob1,
    "ScrewdriverHandoverFnWuang": ScrewdriverHandoverFnWuang,
    "ScrewdriverHandoverFnJoey": ScrewdriverHandoverFnJoey,
    "ScrewdriverHandoverFnGiolle": ScrewdriverHandoverFnGiolle,
    "ScrewdriverPickFnDavide": ScrewdriverPickFnDavide,
    "ScrewdriverHandoverFnDavide": ScrewdriverHandoverFnDavide,
    "ScrewdriverHandoverFnDistanceTableLow": ScrewdriverHandoverFnDistanceTableLow,
    "ScrewdriverPickFnHugo": ScrewdriverPickFnHugo,
    "ScrewdriverHandoverFnDaniele": ScrewdriverHandoverFnDaniele,
    "ScrewdriverHandoverFnDaniele2": ScrewdriverHandoverFnDaniele2,
    "ScrewdriverPickFn11": ScrewdriverPickFn11,
    "ScrewdriverHandoverFn11": ScrewdriverHandoverFn11,
    "ScrewdriverHandoverFn12": ScrewdriverHandoverFn12,
}
