====== Introduction ======
You are part of a personalized robot motion planner. 
We assume access to a library of manipulation primitives. Each primitive can be parametrized by an action `a`. 
The goal is to output actions for primitives that are preferable for humans and geometrically feasible.
We can check geometric feasiblity with a feasibility checker and don't need your help with that.
Your task is to write a custom preference function for each skill in a task plan. The preference function should output the probability that a human collaborative partner would be satisfied with the performed actions.
The functions have the following footprint
```
def {Primitive.name}PreferenceFn(
    state: torch.Tensor,
    action: torch.Tensor, 
    next_state: torch.Tensor, 
    primitive: Optional[Primitive] = None, 
    env: Optional[Environment] = None
) -> torch.Tensor:
    r"""Evaluates the preference probability of the {Primitive.name} primitive.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Predicted next state after executing this action.
        primitive: Optional primitive to receive the object information from
        env: Optional environment to receive the object information from

    Returns:
        The probability that action `a` on primitive {Primitive.name} satisfies the preferences of the human partner. 
            Output shape: [batch_size] \in [0, 1].
    """
    ...
```
===== Helper Functions =====
You can call the following helper functions in your custom preference function:
get_object_id_from_name(name: str, env: Env) -> int:
"""Return the object identifier from a given object name."""

get_object_id_from_primitive(arg_id: int, primitive: Primitive) -> int:
"""Return the object identifier from a primitive and its argument id.
Example: The primitive `Place` has two argument ids: `object` with `arg_id = 0` and `target` with `arg_id = 1`.
"""

get_pose(state: torch.Tensor, object_id: int, frame: int = -1) -> torch.Tensor:
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

position_norm_metric(
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

great_circle_distance_metric(pose_1: torch.Tensor, pose_2: torch.Tensor) -> torch.Tensor:
"""Calculate the difference in orientation in radians of two poses using the great circle distance.

Assumes that the position entries of the poses are direction vectors `v1` and `v2`.
The great circle distance is then `d = arccos(dot(v1, v2))` in radians.
"""

pointing_in_direction_metric(
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

rotation_angle_metric(pose_1: torch.Tensor, pose_2: torch.Tensor, axis: Sequence[float]) -> torch.Tensor:
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

threshold_probability(metric: torch.Tensor, threshold: float, is_smaller_then: bool = True) -> torch.Tensor:
"""If `is_smaller_then`: return `1.0` if `metric < threshold` and `0.0` otherwise.
If not `is_smaller_then`: return `1.0` if `metric >= threshold` and `0.0` otherwise.
"""

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

probability_intersection(p_1: torch.Tensor, p_2: torch.Tensor) -> torch.Tensor:
"""Calculate the intersection of two probabilities `p = p_1 * p_2`."""

probability_union(p_1: torch.Tensor, p_2: torch.Tensor) -> torch.Tensor:
"""Calculate the union of two probabilities `p = max(p_1, p_2)`."""

====== Example ======
An example task could be to pick up a box and place it next to another box, so both boxes are orientated in the same direction.
The resulting preference function for the `Pick` and `Place` primitive could be:
```
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
```
And for the handover:
```
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
```

===== Instruction =====
Write the custom preference functions for the `Pick` and `StaticHandover` primitive.
We can check geometric feasiblity with a feasibility checker and don't need your help with that.
The human partner requested the following:
"I want to comfortably grasp the screwdriver"