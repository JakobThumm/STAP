import pathlib
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from stap import agents, dynamics, envs
from stap.utils import configs


class DynamicsFactory(configs.Factory):
    """Dynamics factory."""

    def __init__(
        self,
        config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        policies: Optional[Sequence[agents.Agent]] = None,
        policy_checkpoints: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        env_kwargs: Dict[str, Any] = {},
        env: Optional[envs.Env] = None,
        device: str = "auto",
    ):
        """Creates the dynamics model factory from a config or checkpoint.

        Args:
            config: Optional dynamics config path or dict. Must be provided if
                checkpoint is None.
            checkpoint: Optional dynamics checkpoint path. Must be provided if
                config is None.
            policies: Optional list of dynamics policies. Must be provided if
                policy_checkpoints is None.
            policy_checkpoints: Optional list of policy checkpoints. Must be
                provided if policies is None.
            env_kwargs: Kwargs passed to EnvFactory for each policy checkpoint.
            env: Env required only for OracleDynamics and TableEnvDynamics.
            device: Torch device.
        """
        if checkpoint is not None:
            ckpt_config = load_config(checkpoint)
            if config is None:
                config = ckpt_config
            if policy_checkpoints is None:
                policy_checkpoints = load_policy_checkpoints(checkpoint)

        if config is None:
            raise ValueError("Either config or checkpoint must be specified")

        if policies is None and policy_checkpoints is not None:
            policies = [
                agents.load(checkpoint=policy_checkpoint, env_kwargs=env_kwargs)
                for policy_checkpoint in policy_checkpoints
            ]

        if policies is None:
            raise ValueError("One of config, policies, or policy_checkpoints must be specified")

        super().__init__(config, "dynamics", dynamics)

        if checkpoint is not None and self.config["dynamics"] != ckpt_config["dynamics"]:
            raise ValueError(
                f"Config dynamics [{self.config['dynamics']}] and checkpoint"
                f"dynamics [{ckpt_config['dynamics']}] must be the same"
            )

        if issubclass(self.cls, dynamics.LatentDynamics):
            self.kwargs["checkpoint"] = checkpoint

        if issubclass(self.cls, (dynamics.OracleDynamics, dynamics.TableEnvDynamics)):
            if issubclass(self.cls, dynamics.OracleDynamics) and env is None:
                raise ValueError(f"{self.cls} requires env to be specified")
            self.kwargs["env"] = env

        self.kwargs["device"] = device
        self.kwargs["policies"] = policies

        self._policy_checkpoints = policy_checkpoints

    def save_config(self, path: Union[str, pathlib.Path]) -> None:
        """Saves the config to path.

        Args:
            path: Directory where config will be saved.
        """
        super().save_config(path)
        if self._policy_checkpoints is None:
            return

        path = pathlib.Path(path)
        with open(path / "policy_checkpoints.txt", "w") as f:
            f.write("\n".join(map(str, self._policy_checkpoints)))


def load(
    config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    checkpoint: Optional[Union[str, pathlib.Path]] = None,
    policies: Optional[Sequence[agents.Agent]] = None,
    policy_checkpoints: Optional[Sequence[Union[str, pathlib.Path]]] = None,
    env_kwargs: Dict[str, Any] = {},
    env: Optional[envs.Env] = None,
    device: str = "auto",
    **kwargs,
) -> dynamics.Dynamics:
    """Loads the dynamics model from a config or checkpoint.

    Args:
        config: Optional dynamics config path or dict. Must be provided if
            checkpoint is None.
        checkpoint: Optional dynamics checkpoint path. Must be provided if
            config is None.
        policies: Optional list of dynamics policies. Must be provided if
            policy_checkpoints is None.
        policy_checkpoints: Optional list of policy checkpoints. Must be
            provided if policies is None.
        env_kwargs: Kwargs passed to EnvFactory for each policy checkpoint.
        env: Env required only for OracleDynamics.
        device: Torch device.
        kwargs: Optional dynamics constructor kwargs.

    Returns:
        Dynamics instance.
    """
    dynamics_factory = DynamicsFactory(
        config=config,
        checkpoint=checkpoint,
        policies=policies,
        policy_checkpoints=policy_checkpoints,
        env_kwargs=env_kwargs,
        env=env,
        device=device,
    )
    return dynamics_factory(**kwargs)


def load_config(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Loads a dynamics config from path.

    Args:
        path: Path to the config, config directory, or checkpoint.

    Returns:
        Dynamics config dict.
    """
    return configs.load_config(path, "dynamics")


def load_policy_checkpoints(path: Union[str, pathlib.Path]) -> List[pathlib.Path]:
    """Loads a dynamics config from path.

    Args:
        path: Path to the config, config directory, or checkpoint.

    Returns:
        Dynamics config dict.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)

    if path.name == "policy_checkpoints.txt":
        policy_checkpoints_path = path
    else:
        if path.suffix == ".pt":
            path = path.parent

        policy_checkpoints_path = path / "policy_checkpoints.txt"

    with open(policy_checkpoints_path, "r") as f:
        policy_checkpoints = [pathlib.Path(line.rstrip()) for line in f]

    return policy_checkpoints


def batch_rotations_6D_to_matrix(batch: torch.Tensor) -> torch.Tensor:
    """Converts a batch of 6D rotations into rotation matrices.

    Args:
        batch: Batch of rotations in shape [B, H, 6], where each rotation is
            represented by the entries R11, R21, R31, R21, R22, R23.

    Returns:
        A tensor of rotation matrices with shape [B, H, 3, 3].
    """
    B, H, d = batch.shape
    assert d == 6, f"Expected batch to have shape [{B}, {H}, 6], but got {batch.shape}"

    # Extract a1 and a2 vectors
    a1 = batch[..., :3]  # shape [B, H, 3]
    a2 = batch[..., 3:]  # shape [B, H, 3]

    # Normalize a1 to get b1
    b1 = a1 / a1.norm(dim=-1, keepdim=True)

    # Project a2 onto the orthogonal complement of b1 to get u2, then normalize u2 to get b2
    dot_product = (a2 * b1).sum(dim=-1, keepdim=True)
    u2 = a2 - dot_product * b1  # Remove component of a2 along b1
    b2 = u2 / u2.norm(dim=-1, keepdim=True)

    # Compute b3 as the cross product of b1 and b2
    b3 = torch.cross(b1, b2, dim=-1)  # Make sure the cross product is taken along the correct dim

    # Stack b1, b2, and b3 to form the rotation matrices
    rotations = torch.stack([b1, b2, b3], dim=-1)  # [B, H, 3, 3], stack along the last dimension

    return rotations


def batch_axis_angle_to_matrix(batch: torch.Tensor) -> torch.Tensor:
    """Converts a batch of axis-angle rotations into rotation matrices.

    Axis-angle representation is a 3D vector where the direction of the vector and its magnitude represent the angle.

    Args:
        batch: Batch of rotations in shape [B, H, 3], where each rotation is
            represented by the axis-angle vector.
    Returns:
        A tensor of rotation matrices with shape [B, H, 3, 3].
    """
    B, H, d = batch.shape
    assert d == 3, f"Expected batch to have shape [{B}, {H}, 3], but got {batch.shape}"

    # Unpack the axis-angle representation into the angle and axis
    angle = batch.norm(dim=-1, keepdim=True)  # Shape [B, H, 1]
    axis = batch / angle  # Shape [B, H, 3]

    # Compute the skew-symmetric cross product matrix of the axis
    skew_symmetric = torch.zeros(B, H, 3, 3, device=batch.device, dtype=batch.dtype)
    skew_symmetric[..., 0, 1] = -axis[..., 2]
    skew_symmetric[..., 0, 2] = axis[..., 1]
    skew_symmetric[..., 1, 2] = -axis[..., 0]
    skew_symmetric[..., 1, 0] = axis[..., 2]
    skew_symmetric[..., 2, 0] = -axis[..., 1]
    skew_symmetric[..., 2, 1] = axis[..., 0]

    # Compute the rotation matrix using the Rodrigues' formula
    rotation_matrix = (
        torch.eye(3, device=batch.device, dtype=batch.dtype).unsqueeze(0).unsqueeze(0)
        + skew_symmetric * torch.sin(angle)
        + torch.matmul(skew_symmetric, skew_symmetric) * (1 - torch.cos(angle))
    )

    return rotation_matrix


def matrix_to_6D_rotations(matrix: torch.Tensor) -> torch.Tensor:
    """Converts a batch of rotation matrices into 6D representation.

    Args:
        matrix: Batch of rotation matrices with shape [B, H, 3, 3].
    Returns:
        A tensor of 6D rotations with shape [B, H, 6].
    """
    return matrix.transpose(-1, -2).flatten(start_dim=2)[..., 0:6]


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Converts a batch of rotation matrices into axis-angle representation.

    Args:
        matrix: Batch of rotation matrices with shape [B, H, 3, 3].
    Returns:
        A tensor of axis-angle representations with shape [B, H, 3].
    """
    B, H, d1, d2 = matrix.shape
    assert d1 == 3 and d2 == 3, f"Expected batch to have shape [{B}, {H}, 3, 3], but got {matrix.shape}"

    # Unpack the rotation matrix into the skew-symmetric matrix
    skew_symmetric = matrix - matrix.transpose(-1, -2)  # [B, H, 3, 3]

    # Compute the axis-angle representation using the Rodrigues' formula
    angle = torch.acos((matrix.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2)  # [B, H]
    axis = torch.stack(
        [
            skew_symmetric[..., 2, 1],
            skew_symmetric[..., 0, 2],
            skew_symmetric[..., 1, 0],
        ],
        dim=-1,
    )  # [B, H, 3]

    # Normalize the axis-angle representation
    axis = axis / axis.norm(dim=-1, keepdim=True)  # [B, H, 3]

    return angle * axis


def batch_rotations_6D_squashed_to_matrix(batch: torch.Tensor, H: int) -> torch.Tensor:
    """Converts a batch of 6D rotations into rotation matrices.

    Args:
        batch: Batch of rotations in shape [B, H*6], where each rotation is
            represented by the entries R11, R12, R21, R22, R31, R32.
        H: The number of 6D rotations per batch element.

    Returns:
        A tensor of rotation matrices with shape [B, H, 3, 3].
    """
    B, Z = batch.shape
    assert Z == H * 6, f"Expected batch to have shape [{B}, {H*6}], but got {batch.shape}"
    # Reshape to [B, H, 6] to separate each rotation
    batch = batch.view(B, H, 6)
    return batch_rotations_6D_to_matrix(batch)


def geodesic_loss(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Calculates the geodesic loss between two batches of rotation matrices.

    Args:
        R1: A tensor of rotation matrices with shape [B, H, 3, 3].
        R2: A tensor of rotation matrices with shape [B, H, 3, 3].

    Returns:
        A tensor containing the geodesic loss for each rotation matrix pair, with shape [B, H].
    """
    # Calculate the relative rotation matrix R_p * R_gt^T
    relative_rotation = torch.matmul(R1, R2.transpose(-1, -2))

    # Calculate the trace of each relative rotation matrix
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1)  # Sum along the last dimension to get the trace

    # Compute the geodesic loss using the provided formula
    # Ensuring the value passed to arccos is within its domain
    # Ensure loss doesn't go to infinity.
    t = (trace - 1.0) / 2.0
    t_clamp = t + (torch.clamp(t, -0.98, 0.98) - t).detach()
    loss = torch.acos(t_clamp)

    return loss
