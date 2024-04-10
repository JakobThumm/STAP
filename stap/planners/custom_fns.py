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


def position_diff_along_direction(
    pose_1: torch.Tensor, pose_2: torch.Tensor, direction: Sequence[float] = [1, 0, 0]
) -> torch.Tensor:
    """Calculate the positional difference of two poses along the given direction.

    Can be used to evaluate if an object is placed left or right of another object.

    Args:
        pose_{1, 2}: the poses of the two objects.
        direction: the direction along which to calculate the difference.
    Returns:
        The positional difference in shape [..., 1]
    """
    direction = torch.FloatTensor(direction).to(pose_1.device)  # type: ignore
    direction = direction / torch.norm(direction, dim=-1, keepdim=True)  # type: ignore
    position_diff = pose_1[..., :3] - pose_2[..., :3]
    return torch.sum(position_diff * direction, dim=-1, keepdim=True)


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
    red_box_id = get_object_id_from_name("red_box", env)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed left of the red box
    left = [0.0, 1.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, left)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed left of the red box.
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate if the object has a deviation in the x direction.
    x_diff_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x"])
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
    blue_box_id = get_object_id_from_name("blue_box", env)
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed left of the blue box
    left = [-1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, left)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed left of the blue box.
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate if the object has a deviation in the x direction.
    y_diff_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["y"])
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
    blue_box_id = get_object_id_from_name("blue_box", env)
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


CUSTOM_FNS = {
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
}
