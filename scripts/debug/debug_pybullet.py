#!/usr/bin/env python3

import argparse
from typing import Optional

import numpy as np

from stap import envs
from stap.envs import pybullet


def main(env_config: str, seed: Optional[int] = None) -> None:
    env_factory = envs.EnvFactory(config=env_config)
    env = env_factory()
    assert isinstance(env, pybullet.table_env.TableEnv)

    while True:
        _, info = env.reset(seed=seed)
        seed = None

        print("Reset seed:", info["seed"])

        action_skeleton = env.task.action_skeleton
        for step in range(len(action_skeleton)):
            # Set and get primitive
            env.set_primitive(primitive=action_skeleton[step])
            primitive = env.get_primitive()
            assert isinstance(primitive, pybullet.table.primitives.Primitive)
            print(f"Execute primitive: {primitive}")

            # Sample action and step environment
            action = primitive.sample_action()
            if isinstance(primitive, pybullet.table.primitives.Handover):
                action.pitch = -np.pi / 2
                action.distance = 0.1
            normalized_action = primitive.normalize_action(action.vector)
            _, success, _, truncated, _ = env.step(normalized_action)
            print(f"Success {primitive}: {success}")

            if truncated:
                break

        print("Done task, continue?\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-config",
        "-e",
        type=str,
        required=True,
        help="Path to environment config.",
    )
    parser.add_argument("--seed", "-s", type=int, help="Seed to reset env")
    args = parser.parse_args()
    main(**vars(args))
    main(**vars(args))
