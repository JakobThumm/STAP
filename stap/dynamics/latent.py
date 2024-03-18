import pathlib
from typing import Any, Dict, Optional, OrderedDict, Sequence, Tuple, Type, Union

import gym
import numpy as np
import torch

from stap import agents, networks
from stap.dynamics.base import Dynamics
from stap.dynamics.utils import geodesic_loss
from stap.utils import configs
from stap.utils.transformation_utils import (
    batch_rotations_6D_squashed_to_matrix,
    rotation_6d_to_matrix,
)
from stap.utils.typing import DynamicsBatch, Model, Scalar


class LatentDynamics(Dynamics, Model[DynamicsBatch]):
    """Base dynamics class."""

    def __init__(
        self,
        policies: Sequence[agents.RLAgent],
        network_class: Union[str, Type[networks.dynamics.Dynamics]],
        network_kwargs: Dict[str, Any],
        state_space: Optional[gym.spaces.Box] = None,
        action_space: Optional[gym.spaces.Box] = None,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
        has_6D_rot_state: bool = True,
        rot_pos_in_state: int = 3,
        use_geodesic_loss: bool = True,
        geodesic_loss_factor: float = 0.01,
    ):
        """Initializes the dynamics model network, dataset, and optimizer.

        Args:
            policies: Ordered list of all policies.
            network_class: Dynamics model network class.
            network_kwargs: Kwargs for network class.
            state_space: Optional state space.
            action_space: Optional action space.
            checkpoint: Dynamics checkpoint.
            device: Torch device.
            has_6D_rot_state: Whether the state space includes 6D rotation or the 3D axis-angle representation.
            rot_pos_in_state: Position of the first entry of the rotation in the state space.
            use_geodesic_loss: Whether to use the geodesic loss for the rotation.
            geodesic_loss_factor: Factor to multiply the geodesic loss by.
        """
        network_class = configs.get_class(network_class, networks)
        self._network = network_class(**network_kwargs)
        self._has_6D_rot_state = has_6D_rot_state
        self._rot_pos_in_state = rot_pos_in_state
        self._use_geodesic_loss = use_geodesic_loss
        self._geodesic_loss_factor = geodesic_loss_factor

        super().__init__(
            policies=policies,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        if checkpoint is not None:
            self.load(checkpoint, strict=True)

    @property
    def network(self) -> networks.dynamics.Dynamics:
        """Dynamics model network."""
        return self._network

    def load_state_dict(self, state_dict: Dict[str, OrderedDict[str, torch.Tensor]], strict: bool = True):
        """Loads the dynamics state dict.

        Args:
            state_dict: Torch state dict.
            strict: Ensure state_dict keys match networks exactly.
        """
        self.network.load_state_dict(state_dict["dynamics"], strict=strict)

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Gets the dynamics state dict."""
        return {
            "dynamics": self.network.state_dict(),
        }

    def create_optimizers(
        self,
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any],
    ) -> Dict[str, torch.optim.Optimizer]:
        """Creates the optimizers for training.

        This method is called by the Trainer class.

        Args:
            optimizer_class: Optimizer class.
            optimizer_kwargs: Kwargs for optimizer class.
        Returns:
            Dict of optimizers.
        """
        optimizers = {"dynamics": optimizer_class(self.network.parameters(), **optimizer_kwargs)}
        return optimizers

    def to(self, device: Union[str, torch.device]) -> "LatentDynamics":
        """Transfers networks to device."""
        super().to(device)
        self.network.to(self.device)
        return self

    def train_mode(self) -> None:
        """Switches to training mode."""
        self.network.train()

    def eval_mode(self) -> None:
        """Switches to eval mode."""
        self.network.eval()

    def plan_mode(self) -> None:
        """Switches to plan mode."""
        self.eval_mode()

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        idx_policy: Union[int, torch.Tensor],
        policy_args: Union[np.ndarray, Optional[Any]],
    ) -> torch.Tensor:
        """Predicts the next latent state given the current latent state and
        action.

        Args:
            state: Current latent state.
            action: Policy action.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Prediction of next latent state.
        """
        dz = self.network(state, idx_policy, action)
        return state + dz

    def get_mse_and_geodesic_loss_entries(self, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the position of the entries that are to be used for the MSE and the geodesic loss.

        The original batch has the following shape: [B, H, W].
        That is converted in the encoder to a latent state with shape: [B, Z], where Z = H * W, Z = [W_1, ..., W_H].
        The Geodesic loss is computed only for the rotation part of the latent state.
        The rotation is located in W at positions:
            [rot_pos_in_state : rot_pos_in_state + 3], if has_6D_rot_state = False
            [rot_pos_in_state : rot_pos_in_state + 6], if has_6D_rot_state = True
        The MSE loss is located in W at positions [0, rot_pos_in_state] + :
            [rot_pos_in_state + 3 :], if has_6D_rot_state = False
            [rot_pos_in_state + 6 :], if has_6D_rot_state = True

        Args:
            H: Height of the latent state.
            W: Width of the latent state.
        Returns:
            Indices for the MSE loss and the geodesic loss in latent space (Z).
        """

        # Calculating the index range for rotation in latent state
        rot_end = self._rot_pos_in_state + 6 if self._has_6D_rot_state else self._rot_pos_in_state + 3
        base_mse_indices = torch.cat((torch.arange(0, self._rot_pos_in_state), torch.arange(rot_end, W)))
        base_geodesic_indices = torch.arange(self._rot_pos_in_state, rot_end)

        # Repeat indices for each "layer" in H
        mse_indices = torch.cat([base_mse_indices + i * W for i in range(H)]).long()
        geodesic_indices = torch.cat([base_geodesic_indices + i * W for i in range(H)]).long()

        return mse_indices, geodesic_indices

    def _unnormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        # Unflatten state if planning.
        raise NotImplementedError("This function has to be implemented in a parent class.")

    def compute_loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        idx_policy: torch.Tensor,
        policy_args: np.ndarray,
    ) -> Tuple[torch.Tensor, Dict[str, Union[Scalar, np.ndarray]]]:
        """Computes the L2 loss between the predicted next latent and the latent
        encoded from the given next observation.

        Args:
            observation: Common observation across all policies.
            action: Policy parameters.
            next_observation: Next observation.
            idx_policy: Index of executed policy.

        Returns:
            L2 loss.
        """
        # Predict next latent state.
        # [B, H, W], [B] => [B, Z].
        latent = self.encode(observation, idx_policy, policy_args)

        # [B, Z], [B], [B, A] => [B, Z].
        next_latent_pred = self.forward(latent, action, idx_policy, policy_args)

        # Encode next latent state.
        # [B, H, W], [B] => [B, Z].
        next_latent = self.encode(next_observation, idx_policy, policy_args)

        # Compute L2 loss.
        # [B, Z], [B, Z] => [1].
        # Get the indices of the rotation and everything else.
        if self._use_geodesic_loss:
            mse_indices, geodesic_indices = self.get_mse_and_geodesic_loss_entries(
                observation.shape[1], observation.shape[2]
            )
            # MSE Loss
            mse_loss = torch.nn.functional.mse_loss(next_latent_pred[:, mse_indices], next_latent[:, mse_indices])
            if self._has_6D_rot_state:
                next_obs_predicted = self._unnormalize_state(next_latent_pred)
                predicted_R = batch_rotations_6D_squashed_to_matrix(
                    next_obs_predicted[:, geodesic_indices], observation.shape[1]
                )
                true_R = rotation_6d_to_matrix(next_observation[:, :, 3:9])
            else:
                raise NotImplementedError("Only 6D rotations are supported.")
            rotational_losses = geodesic_loss(predicted_R, true_R)
            rotational_loss = rotational_losses.mean()
            total_loss = mse_loss + self._geodesic_loss_factor * rotational_loss
        else:
            mse_loss = torch.nn.functional.mse_loss(next_latent_pred, next_latent)
            rotational_loss = torch.tensor(0.0)
            total_loss = mse_loss

        metrics: Dict[str, Union[Scalar, np.ndarray]] = {
            "mse_loss": mse_loss.item(),
            "rotational_loss": rotational_loss.item(),
            "loss": total_loss.item(),
        }
        if self._use_geodesic_loss:
            for i_policy, _ in enumerate(self.policies):
                metrics[f"geodesic_loss_{i_policy}"] = rotational_losses[i_policy == idx_policy].mean().item()

        """This is only for computing metrics on the individual policies. It is not used for training.
        # Compute per-policy L2 losses.
        # [B, Z], [B, Z] => [B].
        l2_losses = torch.nn.functional.mse_loss(next_latent_pred, next_latent, reduction="none").mean(dim=-1)

        # [B], [P] => [B, P].
        idx_policies = idx_policy.unsqueeze(-1) == torch.arange(len(self.policies), device=self.device)

        # [B] => [B, P].
        l2_losses = l2_losses.unsqueeze(-1).tile((len(self.policies),))

        # [B] => [B, P].
        policy_l2_losses = l2_losses * idx_policies

        # [B, P], [B, P] => [P].
        batch_dims = list(range(len(l2_losses.shape) - 1))
        policy_l2_losses = policy_l2_losses.sum(dim=batch_dims) / idx_policies.sum(dim=batch_dims)

        for i_policy, policy_l2_loss in enumerate(policy_l2_losses.detach().cpu().numpy()):
            metrics[f"l2_loss_policy_{i_policy}"] = policy_l2_loss
        """
        return total_loss, metrics

    def train_step(
        self,
        step: int,
        batch: DynamicsBatch,
        optimizers: Dict[str, torch.optim.Optimizer],
        schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler],
    ) -> Dict[str, Union[Scalar, np.ndarray]]:
        """Performs a single training step.

        Args:
            step: Training step.
            batch: Training batch.
            optimizers: Optimizers created in `LatentDynamics.create_optimizers()`.
            schedulers: Schedulers with the same keys as `optimizers`.

        Returns:
            Dict of training metrics for logging.
        """
        loss, metrics = self.compute_loss(**batch)  # type: ignore

        optimizers["dynamics"].zero_grad()
        loss.backward()
        optimizers["dynamics"].step()
        schedulers["dynamics"].step()

        return metrics
