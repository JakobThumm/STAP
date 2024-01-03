"""Defines custom evaluation functions to be used in the trajectory evaluation of the planner.

Author: Jakob Thumm
Date: 2024-01-02
"""

from typing import Optional

import torch

from stap import envs
from stap.envs.pybullet.table_env import TableEnv


def HookHandoverOrientationFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, env: Optional[envs.Env] = None
) -> torch.Tensor:
    r"""Evaluates the orientation of the hook handover.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        env: optional environment to call functions

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    assert env is not None
    assert isinstance(env, TableEnv)
    object_orientation = env.get_object_orientation_from_observation(next_state, "hook")
    MIN_VALUE = 0.5
    MAX_VALUE = 1.0
    OPTIMAL_ORIENTATION = torch.pi / 2
    return_value = MIN_VALUE + (torch.abs(object_orientation[:, 2] - OPTIMAL_ORIENTATION)) / (2 * torch.pi) * (
        MAX_VALUE - MIN_VALUE
    )
    return return_value


CUSTOM_FNS = {"HookHandoverOrientationFn": HookHandoverOrientationFn}
