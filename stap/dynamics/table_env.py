import pathlib
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import gym
import numpy as np
import torch

from stap import agents, envs, networks
from stap.dynamics.latent import LatentDynamics
from stap.dynamics.utils import (
    batch_axis_angle_to_matrix,
    batch_rotations_6D_to_matrix,
    matrix_to_6D_rotations,
    matrix_to_axis_angle,
)
from stap.envs.base import Primitive
from stap.envs.pybullet.table.objects import Rack
from stap.envs.pybullet.table.primitives import ACTION_CONSTRAINTS
from stap.envs.pybullet.table_env import TableEnv
from stap.utils.transformation_utils import euler_angles_to_matrix


class TableEnvDynamics(LatentDynamics):
    """Dynamics model per action with shared latent states.

    We train A dynamics models T_a of the form:

        z^(t+1) = z^(t) + T_a(z^(t), a^(t))

    for every action a.
    """

    def __init__(
        self,
        policies: Sequence[agents.RLAgent],
        network_class: Union[str, Type[networks.dynamics.PolicyDynamics]],
        network_kwargs: Dict[str, Any],
        env: Optional[envs.pybullet.TableEnv],
        rigid_body: bool = True,
        hand_crafted: bool = False,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
    ):
        """Initializes the dynamics model network, dataset, and optimizer.

        Args:
            policies: Ordered list of all policies.
            network_class: Backend network for decoupled dynamics network.
            network_kwargs: Kwargs for network class.
            env: TableEnv required for planning (not training).
            rigid_body: Only predict object poses during evaluation.
            hand_crafted: Support dynamics prediction with some manual settings.
            checkpoint: Dynamics checkpoint.
            device: Torch device.
        """
        self._env = env
        self._plan_mode = False
        self._rigid_body = rigid_body
        self._hand_crafted = hand_crafted

        if self.env is None:
            observation_space = policies[0].observation_space
        else:
            observation_space = self.env.observation_space

        self._observation_mid = torch.from_numpy((observation_space.low[0] + observation_space.high[0]) / 2)
        self._observation_range = torch.from_numpy(observation_space.high[0] - observation_space.low[0])

        self._observation_space = observation_space
        flat_observation_space = gym.spaces.Box(
            low=observation_space.low.flatten(),
            high=observation_space.high.flatten(),
        )
        self._flat_state_space = gym.spaces.Box(
            low=-0.5,
            high=0.5,
            shape=flat_observation_space.shape,
            dtype=flat_observation_space.dtype,  # type: ignore
        )

        parent_network_class = networks.dynamics.Dynamics
        parent_network_kwargs = {
            "policies": policies,
            "network_class": network_class,
            "network_kwargs": network_kwargs,
            "state_spaces": [self.flat_state_space] * len(policies),
        }
        super().__init__(
            policies=policies,
            network_class=parent_network_class,
            network_kwargs=parent_network_kwargs,
            state_space=self.state_space,
            action_space=None,
            checkpoint=checkpoint,
            device=device,
        )

    @property
    def env(self) -> Optional[envs.pybullet.TableEnv]:
        return self._env

    @property
    def state_space(self) -> gym.spaces.Box:
        if self._plan_mode:
            return self._observation_space
        else:
            return self._flat_state_space

    @property
    def flat_state_space(self) -> gym.spaces.Box:
        return self._flat_state_space

    def to(self, device: Union[str, torch.device]) -> LatentDynamics:
        """Transfers networks to device."""
        super().to(device)
        self._observation_mid = self._observation_mid.to(self.device)
        self._observation_range = self._observation_range.to(self.device)
        return self

    def train_mode(self) -> None:
        """Switches to train mode."""
        super().train_mode()
        self._plan_mode = False

    def eval_mode(self) -> None:
        """Switches to eval mode."""
        super().eval_mode()
        self._plan_mode = False

    def plan_mode(self) -> None:
        """Switches to plan mode."""
        super().eval_mode()
        self._plan_mode = True

    def encode(
        self,
        observation: torch.Tensor,
        idx_policy: Union[int, torch.Tensor],
        policy_args: Union[np.ndarray, Optional[Dict[str, List[int]]]],
    ) -> torch.Tensor:
        """Encodes the observation into a dynamics state.

        During training, the dynamics state is equivalent to the policy state
        (normalized vector containing state for 3 objects) appended with
        additional object states. During planning, the dynamics state is
        equivalent to the environment observation (unnormalized matrix
        containing state for all objects).

        Args:
            observation: Common observation across all policies.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Encoded latent state vector.
        """
        if self._plan_mode:
            # Return full observation.
            return observation

        assert policy_args is not None
        observation = networks.encoders.TableEnvEncoder.rearrange_observation(observation, policy_args, randomize=False)

        dynamics_state = self._normalize_state(observation)

        return dynamics_state

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        # Scale to [-0.5, 0.5].
        state = (state - self._observation_mid) / self._observation_range

        # Flatten state.
        if state.ndim > len(self.state_space.shape):
            state = state.reshape(-1, *self.flat_state_space.shape)
        else:
            state = state.reshape(*self.flat_state_space.shape)

        return state

    def _unnormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        # Unflatten state if planning.
        state = state.reshape(-1, *self.state_space.shape)
        if len(state.size()) == 2 and state.size(1) > self._observation_range.size(0):
            assert state.size(1) % self._observation_range.size(0) == 0
            N = state.size(1) // self._observation_range.size(0)
            observation_range = self._observation_range.repeat(N)
            observation_mid = self._observation_mid.repeat(N)
        else:
            observation_range = self._observation_range
            observation_mid = self._observation_mid
        # Scale from [-0.5, 0.5].
        state = state * observation_range + observation_mid

        return state

    def decode(self, state: torch.Tensor, primitive: envs.Primitive) -> torch.Tensor:
        """Decodes the dynamics state into policy states.

        This is only used during planning, not training, so the input state will
        be the environment state.

        Args:
            state: Full TableEnv observation.
            primitive: Current primitive.

        Returns:
            Decoded observation.
        """
        return self.policies[primitive.idx_policy].encoder.encode(state, policy_args=primitive.get_policy_args())

    def _apply_handcrafted_dynamics(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        predicted_next_state: torch.Tensor,
        primitive: envs.Primitive,
        policy_args: Optional[Dict[str, List[int]]],
    ) -> torch.Tensor:
        """Applies handcrafted dynamics to the state.

        Args:
            state: Current state.
            action: Policy action.
            predicted_next_state: Predicted next state (by network)
            primitive: Current primitive.

        Returns:
            Prediction of next state.
        """
        primitive_str = str(primitive).lower()
        if "pick" in primitive_str:
            target_object_idx = policy_args["observation_indices"][1]
            new_predicted_next_state = self._handcrafted_dynamics_pick(
                target_object_idx, state, predicted_next_state, action, primitive
            )
        elif "place" in primitive_str:
            SRC_OBJ_IDX = 1
            DEST_OBJ_IDX = 2
            source_object_idx = policy_args["observation_indices"][SRC_OBJ_IDX]
            destination_object_idx = policy_args["observation_indices"][DEST_OBJ_IDX]
            new_predicted_next_state = self._handcrafted_dynamics_place(
                source_object_idx, destination_object_idx, state, predicted_next_state, action, primitive
            )
        elif "static_handover" in primitive_str:
            SRC_OBJ_IDX = 1
            DEST_OBJ_IDX = 2
            source_object_idx = policy_args["observation_indices"][SRC_OBJ_IDX]
            destination_object_idx = policy_args["observation_indices"][DEST_OBJ_IDX]
            new_predicted_next_state = self._handcrafted_dynamics_static_handover(
                source_object_idx, destination_object_idx, state, predicted_next_state, action, primitive
            )
        else:
            return predicted_next_state

        return new_predicted_next_state

    def _handcrafted_dynamics_pick(
        self,
        target_object_idx: int,
        current_state: torch.Tensor,
        predicted_next_state: torch.Tensor,
        action: torch.Tensor,
        primitive: Primitive,
    ) -> torch.Tensor:
        X_IDX = 0
        Z_IDX = 2
        rotation_IDX_start = 3
        rotation_IDX_end = rotation_IDX_start + 6 if self._has_6D_rot_state else rotation_IDX_start + 3
        assert self.env is not None

        action_range = torch.from_numpy(primitive.action_scale.high - primitive.action_scale.low).to(self.device)
        action_min = torch.from_numpy(primitive.action_scale.low).to(self.device)
        action_0_1 = (action + 1.0) / 2.0
        unnormalized_action = action_0_1 * action_range + action_min

        # Calculate rotations
        theta = unnormalized_action[..., 3]
        if self._has_6D_rot_state:
            current_object_rot_mat = batch_rotations_6D_to_matrix(
                current_state[:, target_object_idx : target_object_idx + 1, rotation_IDX_start:rotation_IDX_end]
            )
        else:
            current_object_rot_mat = batch_axis_angle_to_matrix(
                current_state[:, target_object_idx : target_object_idx + 1, rotation_IDX_start:rotation_IDX_end]
            )
        # Create z-rotation matrix batch with angle theta.
        z_rot_mat = torch.zeros_like(current_object_rot_mat)
        z_rot_mat[:, 0, 0, 0] = torch.cos(theta)
        z_rot_mat[:, 0, 0, 1] = -torch.sin(theta)
        z_rot_mat[:, 0, 1, 0] = torch.sin(theta)
        z_rot_mat[:, 0, 1, 1] = torch.cos(theta)
        z_rot_mat[:, 0, 2, 2] = 1.0
        # Multiply current rotation matrix with z-rotation matrix.
        new_rot_mat = torch.matmul(current_object_rot_mat, z_rot_mat)
        if self._has_6D_rot_state:
            new_rotation_entries = matrix_to_6D_rotations(new_rot_mat)
        else:
            new_rotation_entries = matrix_to_axis_angle(new_rot_mat)
        predicted_next_state[:, target_object_idx : target_object_idx + 1, rotation_IDX_start:rotation_IDX_end] = (
            new_rotation_entries
        )
        predicted_next_state[
            ..., TableEnv.EE_OBSERVATION_IDX : TableEnv.EE_OBSERVATION_IDX + 1, rotation_IDX_start:rotation_IDX_end
        ] = predicted_next_state[..., target_object_idx : target_object_idx + 1, rotation_IDX_start:rotation_IDX_end]

        # Calculate x- and y-position
        # The object stays where it was
        predicted_next_state[..., target_object_idx, X_IDX:Z_IDX] = current_state[..., target_object_idx, X_IDX:Z_IDX]
        # The end effector moves in the x-y plane by the given action
        displacement = unnormalized_action[..., :3]
        displacement[..., 2] = 0
        displacement_rotated = torch.matmul(new_rot_mat[:, 0, ...], displacement.unsqueeze(-1)).squeeze(-1)
        predicted_next_state[..., TableEnv.EE_OBSERVATION_IDX, X_IDX:Z_IDX] = (
            current_state[..., target_object_idx, X_IDX:Z_IDX] + displacement_rotated[..., 0:2]
        )

        # Calculate z-position
        predicted_next_state[..., TableEnv.EE_OBSERVATION_IDX, Z_IDX] = (
            ACTION_CONSTRAINTS["max_lift_height"] + self.env.robot.arm.ee_offset[Z_IDX]
        )
        predicted_next_state[..., target_object_idx, Z_IDX] = (
            ACTION_CONSTRAINTS["max_lift_height"] - unnormalized_action[..., 2]
        )
        return predicted_next_state

    def _handcrafted_dynamics_place(
        self,
        source_object_idx: int,
        destination_object_idx: int,
        current_state: torch.Tensor,
        predicted_next_state: torch.Tensor,
        action: torch.Tensor,
        primitive: Primitive,
    ) -> torch.Tensor:
        primitive_str = str(primitive).lower()
        destination_object_state = predicted_next_state[..., destination_object_idx, :]

        if "table" in primitive_str:
            destination_object_surface_offset = 0
        elif "rack" in primitive_str:
            destination_object_surface_offset = Rack.TOP_THICKNESS
        else:
            return predicted_next_state

        # hardcoded object heights
        if "box" in primitive_str:
            median_object_height = 0.08
        elif "hook" in primitive_str:
            median_object_height = 0.04
        elif "screwdriver" in primitive_str:
            median_object_height = 0.02
        else:
            return predicted_next_state

        predicted_next_state[..., source_object_idx, 2] = (
            destination_object_state[..., 2] + destination_object_surface_offset + median_object_height / 2
        )
        return predicted_next_state

    def _handcrafted_dynamics_static_handover(
        self,
        object_idx: int,
        target_idx: int,
        current_state: torch.Tensor,
        predicted_next_state: torch.Tensor,
        action: torch.Tensor,
        primitive: Primitive,
    ) -> torch.Tensor:
        assert self.env is not None
        assert hasattr(self.env.robot.arm, "base_pos")
        ADDITIONAL_OFFSET = np.array([0, 0, 0.2])
        pos_IDX = 0
        rotation_IDX_start = 3
        rotation_IDX_end = rotation_IDX_start + 6 if self._has_6D_rot_state else rotation_IDX_start + 3
        # Action: [pitch, yaw, distance, height]
        action_range = torch.from_numpy(primitive.action_scale.high - primitive.action_scale.low).to(self.device)
        action_min = torch.from_numpy(primitive.action_scale.low).to(self.device)
        action_0_1 = (action + 1.0) / 2.0
        unnormalized_action = action_0_1 * action_range + action_min
        target_pos = current_state[:, target_idx, pos_IDX : pos_IDX + 3]
        pitch = unnormalized_action[..., 0]
        yaw = unnormalized_action[..., 1]
        distance = unnormalized_action[..., 2]
        height = unnormalized_action[..., 3]
        # Set height
        target_pos[..., 2] = height
        base_pos = self.env.robot.arm.base_pos + ADDITIONAL_OFFSET  # type: ignore
        height = torch.min(target_pos[..., 2] - base_pos[2], distance)

        # Mathematics convention:
        # Phi is the polar angle https://en.wikipedia.org/wiki/Spherical_coordinate_system#/media/File:3D_Spherical_2.svg
        phi = torch.arccos(height / torch.clamp(distance, min=0.1))
        # Theta angle is the azimuthal angle https://en.wikipedia.org/wiki/Spherical_coordinate_system#/media/File:3D_Spherical_2.svg
        theta = torch.arctan2(target_pos[..., 1] - base_pos[1], target_pos[..., 0] - base_pos[0])
        command_pos = torch.zeros_like(target_pos)
        command_pos[..., 0] = base_pos[0] + torch.sin(phi) * distance * torch.cos(theta)
        command_pos[..., 1] = base_pos[1] + torch.sin(phi) * distance * torch.sin(theta)
        command_pos[..., 2] = base_pos[2] + height
        predicted_next_state[..., object_idx, pos_IDX : pos_IDX + 3] = command_pos
        # Calculate resulting rotation
        R_eef_desired = euler_angles_to_matrix(
            torch.stack([yaw, pitch, torch.zeros_like(yaw)], dim=-1), convention="ZYX"
        )
        if self._has_6D_rot_state:
            R_eef_current = batch_rotations_6D_to_matrix(
                current_state[
                    :,
                    TableEnv.EE_OBSERVATION_IDX : TableEnv.EE_OBSERVATION_IDX + 1,
                    rotation_IDX_start:rotation_IDX_end,
                ]
            )
            R_obj_current = batch_rotations_6D_to_matrix(
                current_state[
                    :,
                    object_idx : object_idx + 1,
                    rotation_IDX_start:rotation_IDX_end,
                ]
            )
        else:
            R_eef_current = batch_axis_angle_to_matrix(
                current_state[
                    :,
                    TableEnv.EE_OBSERVATION_IDX : TableEnv.EE_OBSERVATION_IDX + 1,
                    rotation_IDX_start:rotation_IDX_end,
                ]
            )
            R_obj_current = batch_axis_angle_to_matrix(
                current_state[
                    :,
                    object_idx : object_idx + 1,
                    rotation_IDX_start:rotation_IDX_end,
                ]
            )
        R_T = torch.linalg.solve(R_eef_current[:, 0, ...], R_eef_desired, left=False)
        R_obj_next = torch.matmul(R_T, R_obj_current[:, 0, ...])
        if self._has_6D_rot_state:
            rotation_entries = matrix_to_6D_rotations(R_obj_next.unsqueeze(1))
        else:
            rotation_entries = matrix_to_axis_angle(R_obj_next.unsqueeze(1))
        predicted_next_state[..., object_idx : object_idx + 1, rotation_IDX_start:rotation_IDX_end] = rotation_entries
        return predicted_next_state

    def forward_eval(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        primitive: envs.Primitive,
        use_handcrafted_dynamics_primitives: Optional[List[str]] = ["pick", "place", "static_handover"],
    ) -> torch.Tensor:
        """Predicts the next state for planning.

        During planning, the state is an unnormalized matrix with one row for
        each object. This gets transformed into a normalized policy state vector
        according to the current primitive and fed to the dynamics model. The
        row entries in the state corresponding to the objects involved with the
        primitive are updated according to the dynamics prediction.

        Args:
            state: Current state.
            action: Policy action.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.
            use_handcrafted_dynamics_primitives: List of primitives for
                which to use handcrafted dynamics.

        Returns:
            Prediction of next state.
        """
        env_state = state

        # Env state -> dynamics state.
        policy_args = primitive.get_policy_args()
        assert policy_args is not None
        idx_args = policy_args["observation_indices"]
        dynamics_state = self._normalize_state(env_state[..., idx_args, :])

        # Dynamics state -> dynamics state.
        next_dynamics_state = self.forward(dynamics_state, action, primitive.idx_policy, policy_args)
        next_dynamics_state = next_dynamics_state.clamp(-0.5, 0.5)

        # Update env state with new unnormalized observation.
        next_env_state = env_state.clone()
        next_env_state[..., idx_args, :] = self._unnormalize_state(next_dynamics_state)

        # Keep object shape features consistent across time.
        if self._rigid_body:
            idx_feats = self.env.static_feature_indices
            next_env_state[..., idx_feats] = env_state[..., idx_feats]

        # Apply hand-crafted touch-ups to dynamics.
        if self._hand_crafted and use_handcrafted_dynamics_primitives is not None:
            idx_feats = self.env.dynamic_feature_indices

            for primitive_name in use_handcrafted_dynamics_primitives:
                if primitive_name in str(primitive).lower():
                    next_env_state[..., idx_feats] = self._apply_handcrafted_dynamics(
                        env_state.clone(),
                        action,
                        next_env_state.clone(),
                        primitive,
                        policy_args,
                    )[..., idx_feats]
                    break

        # Set states of non existent objects to 0.
        non_existent_obj_start_idx = policy_args["shuffle_range"][1]
        next_env_state[..., non_existent_obj_start_idx:, :] = 0
        return next_env_state
