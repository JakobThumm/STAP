#!/usr/bin/env python3

import argparse
import sys
from typing import Optional

from stap import envs
from stap.envs import pybullet

ask_for_permission = False


def query_yes_no(question: str, default: str = "yes") -> bool:
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


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
            success = False
            while not success:
                # Sample action and step environment
                action = primitive.sample_action()
                print(f"Execute primitive {primitive} with action {action}")
                if ask_for_permission:
                    input("Press Enter to continue...")
                normalized_action = primitive.normalize_action(action.vector)
                _, success, _, truncated, _ = env.step(normalized_action)
                succes_qestion = query_yes_no("Primitive successful?", default="yes")
                success = success and succes_qestion
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
