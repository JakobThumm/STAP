import argparse
import pathlib
import shutil
from functools import partial

import numpy as np
import tqdm

from stap.datasets.replay_buffer import StorageBatch


def batch_axis_angle_to_matrix(angles):
    N = angles.shape[0]
    # Normalize the axis part of the axis-angle representation
    axis = angles / np.linalg.norm(angles, axis=1)[:, np.newaxis]

    # Extract angles (assumed to be in radians)
    theta = np.linalg.norm(angles, axis=1)

    axis[theta == 0] = [1, 0, 0]

    # Components of axis for convenience
    ux, uy, uz = axis[:, 0], axis[:, 1], axis[:, 2]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Identity components, replicated for each entry
    I = np.eye(3).reshape(3, 3, 1).repeat(N, axis=2).transpose(2, 0, 1)

    # Outer product of axis, for each axis vector
    uuT = axis[:, :, np.newaxis] * axis[:, np.newaxis, :]

    # Skew-symmetric matrices
    U = np.zeros((N, 3, 3))
    U[:, 0, 1] = -uz
    U[:, 1, 0] = uz
    U[:, 0, 2] = uy
    U[:, 2, 0] = -uy
    U[:, 1, 2] = -ux
    U[:, 2, 1] = ux

    # Compute rotation matrices using Rodrigues' formula
    R = (
        cos_theta[:, np.newaxis, np.newaxis] * I
        + (1 - cos_theta)[:, np.newaxis, np.newaxis] * uuT
        + sin_theta[:, np.newaxis, np.newaxis] * U
    )

    R_flat = R.reshape(N, 9)

    return R_flat


def convert_to_6D_rotation(checkpoint: StorageBatch) -> StorageBatch:
    """Converts the axis-angle representation to 6D rotation representation.

    Each row of a timestep is an object. The elements 3, 4, and 5 of each row are the axis-angle representation of the
    object's rotation. The first 3 elements are the object's position.

    Args:
        checkpoint: StorageBatch to convert.
    Returns:
        Converted StorageBatch.
    """
    new_observations = np.zeros(
        [checkpoint["observation"].shape[0], checkpoint["observation"].shape[1], checkpoint["observation"].shape[2] + 3]
    )
    new_observations[:, :, :3] = checkpoint["observation"][:, :, :3]
    for i in range(checkpoint["observation"].shape[1]):
        new_observations[:, i, 3:9] = batch_axis_angle_to_matrix(checkpoint["observation"][:, i, 3:6])[
            :, [0, 3, 6, 1, 4, 7]
        ]
    new_observations[:, :, 9:] = checkpoint["observation"][:, :, 6:]
    checkpoint["observation"] = new_observations
    return checkpoint


def convert_dataset(path: pathlib.Path, out_path: pathlib.Path, conversion_fn: partial):
    """Load replay buffer checkpoints from disk, convert them, and save them to new folder.

    Args:
        path: Location of checkpoints.
        out_path: Location to save converted checkpoints.
        conversion_fn: Function to convert the data to new format.

    Returns:
        Number of entries loaded.
    """
    out_path.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = sorted(path.iterdir(), key=lambda f: tuple(map(int, f.stem.split("_")[:-1])))
    pbar = tqdm.tqdm(checkpoint_paths, desc=f"Load {path}", dynamic_ncols=True)
    for checkpoint_path in pbar:
        with open(checkpoint_path, "rb") as f:
            checkpoint: StorageBatch = dict(np.load(f, allow_pickle=True))  # type: ignore
        checkpoint_out = conversion_fn(checkpoint)
        checkpoint_out_file = out_path / f"{checkpoint_path.stem}.npz"
        with open(checkpoint_out_file, "wb") as f:
            np.savez_compressed(f, **checkpoint_out)


def convert_datasets(base_path_in: pathlib.Path, base_path_out: pathlib.Path, conversion_fn: partial):
    """Convert all datasets in a directory.

    Args:
        base_path_in: Location of datasets.
        base_path_out: Location to save converted datasets.
        conversion_fn: Function to convert the data to new format.
    """
    for path in base_path_in.iterdir():
        if path.is_dir():
            convert_dataset(path / "train_data", base_path_out / path.name / "train_data", conversion_fn)


def main():
    # Argparse
    parser = argparse.ArgumentParser(
        description="Converts the axis-angle representation to 6D rotation representation."
    )
    parser.add_argument("--base_path_in", type=str, default="models/datasets", help="Location of datasets.")
    parser.add_argument(
        "--base_path_out", type=str, default="models/datasets_converted", help="Location to save converted datasets."
    )
    args = parser.parse_args()
    base_path_in = pathlib.Path(args.base_path_in)
    base_path_out = pathlib.Path(args.base_path_out)

    # copy all elements in path to base_path_out / path.name except train_data
    shutil.rmtree(base_path_out, ignore_errors=True)
    shutil.copytree(base_path_in, base_path_out, ignore=shutil.ignore_patterns("train_data"))
    convert_datasets(base_path_in, base_path_out, partial(convert_to_6D_rotation))


if __name__ == "__main__":
    main()
