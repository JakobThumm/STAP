#!/usr/bin/env python3

import argparse
import os

from stap import envs
from stap.envs import pybullet


def main(env_config: str, prompt_template: str) -> None:
    env_factory = envs.EnvFactory(config=env_config)
    env = env_factory()
    assert isinstance(env, pybullet.table_env.TableEnv)
    _, info = env.reset(seed=0)
    # Read in prompts/template_prompt.md file to prompt string.
    prompt = ""
    with open(os.environ["STAP_PATH"] + "/" + prompt_template, "r") as f:
        prompt = f.read()
    # Add info["scene_description"] to prompt string.
    prompt += info["scene_description"]
    # Add instructions to prompt
    prompt += "===== Instruction =====\n" + "Write the custom preference functions for the primitives "
    for primitive in env.action_skeleton:
        prompt += str(primitive) + ", "
    prompt = prompt[:-2] + ".\n"
    prompt += (
        "We can check geometric feasiblity with a feasibility checker and don't need your help with that.\n"
        + "The human partner requested the following:\n"
    )
    print(prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-config",
        "-e",
        type=str,
        required=True,
        help="Path to environment config.",
    )
    parser.add_argument(
        "--prompt-template",
        "-p",
        type=str,
        required=True,
        help="Path to prompt template.",
    )
    args = parser.parse_args()
    main(**vars(args))
