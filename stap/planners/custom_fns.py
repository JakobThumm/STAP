"""Defines custom evaluation functions to be used in the trajectory evaluation of the planner.

Author: Jakob Thumm
Date: 2024-01-02
"""

import torch


def HandoverOrientationFn(state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    r"""Evaluates the orientation of the handover.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.

    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    """
    return torch.ones((state.size()[0],), device=state.device)


CUSTOM_FNS = {"HandoverOrientationFn": HandoverOrientationFn}
