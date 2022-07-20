import argparse
import pathlib
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import tqdm

from temporal_policies import envs, planners
from temporal_policies.utils import random, spaces, timing

import eval_pybox2d_planners as pybox2d


def scale_actions(
    actions: np.ndarray,
    env: envs.Env,
    action_skeleton: Sequence[envs.Primitive],
) -> np.ndarray:
    scaled_actions = actions.copy()
    for t, primitive in enumerate(action_skeleton):
        action_dims = primitive.action_space.shape[0]
        scaled_actions[..., t, :action_dims] = spaces.transform(
            actions[..., t, :action_dims],
            from_space=primitive.action_space,
            to_space=primitive.action_scale,
        )

    return scaled_actions


def evaluate_planners(
    config: Union[str, pathlib.Path],
    env_config: Union[str, pathlib.Path, Dict[str, Any]],
    policy_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]],
    dynamics_checkpoint: Optional[Union[str, pathlib.Path]],
    device: str,
    num_eval: int,
    path: Union[str, pathlib.Path],
    grid_resolution: int,
    verbose: bool,
    seed: Optional[int] = None,
) -> None:
    if seed is not None:
        random.seed(seed)

    timer = timing.Timer()

    env = envs.load(env_config)
    planner = planners.load(
        config=config,
        env=env,
        policy_checkpoints=policy_checkpoints,
        dynamics_checkpoint=dynamics_checkpoint,
        device=device,
    )
    path = pathlib.Path(path) / pathlib.Path(config).stem
    path.mkdir(parents=True, exist_ok=True)

    if isinstance(env, envs.pybox2d.Sequential2D):
        action_skeleton = [
            env.get_primitive_info("PlaceRight"),
            env.get_primitive_info("PushLeft"),
        ]
    elif isinstance(env, envs.pybullet.TableEnv):
        action_skeleton = [
            env.get_primitive_info(action_call) for action_call in env.action_skeleton
        ]

    for i in tqdm.tqdm(range(num_eval), f"Evaluate {path.name}", dynamic_ncols=True):
        if seed is not None:
            random.seed(i)

        observation = env.reset()
        assert isinstance(observation, np.ndarray)
        state = env.get_state()

        timer.tic("planner")
        actions, p_success, visited_actions, p_visited_success = planner.plan(
            observation, action_skeleton
        )
        t_planner = timer.toc("planner")

        rewards = planners.evaluate_plan(
            env, action_skeleton, state, actions, gif_path=path / f"exec_{i}.gif"
        )

        if verbose:
            print("success:", rewards.prod())
            print("predicted success:", p_success)
            print(actions)
            print("time:", t_planner)

        if isinstance(env, envs.pybox2d.Sequential2D):
            env.set_state(state)
            (grid_q_values, grid_actions) = pybox2d.evaluate_critic_functions(
                planner, action_skeleton, env, grid_resolution
            )
            pybox2d.plot_critic_functions(
                env=env,
                action_skeleton=action_skeleton,
                actions=visited_actions,
                p_success=p_visited_success,
                rewards=rewards,
                grid_q_values=grid_q_values,
                grid_actions=grid_actions,
                path=path / f"values_{i}.png",
                title=f"{pathlib.Path(config).stem}: {t_planner:0.2f}s",
            )

        with open(path / f"results_{i}.npz", "wb") as f:
            save_dict = {
                "args": {
                    "config": config,
                    "env_config": env_config,
                    "policy_checkpoints": policy_checkpoints,
                    "dynamics_checkpoint": dynamics_checkpoint,
                    "device": device,
                    "num_eval": num_eval,
                    "path": path,
                    "seed": seed,
                    "grid_resolution": grid_resolution,
                    "verbose": verbose,
                },
                "observation": observation,
                "state": state,
                "actions": actions,
                "scaled_actions": scale_actions(actions, env, action_skeleton),
                "p_success": p_success,
                "rewards": rewards,
                "visited_actions": visited_actions,
                "scaled_visited_actions": scale_actions(
                    visited_actions, env, action_skeleton
                ),
                "p_visited_success": p_visited_success,
                "t_planner": t_planner,
            }
            if isinstance(env, envs.pybox2d.Sequential2D):
                save_dict["grid_q_values"] = grid_q_values
                save_dict["grid_actions"] = grid_actions
            np.savez_compressed(f, **save_dict)  # type: ignore


def main(args: argparse.Namespace) -> None:
    evaluate_planners(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "--planner-config", "--planner", "-c", help="Path to planner config"
    )
    parser.add_argument("--env-config", "--env", "-e", help="Path to env config")
    parser.add_argument(
        "--policy-checkpoints", "-p", nargs="+", help="Policy checkpoints"
    )
    parser.add_argument("--dynamics-checkpoint", "-d", help="Dynamics checkpoint")
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument(
        "--num-eval", "-n", type=int, default=1, help="Number of eval iterations"
    )
    parser.add_argument("--path", default="plots", help="Path for output plots")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=40,
        help="Resolution of critic function plot",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Print debug messages")
    args = parser.parse_args()

    main(args)
