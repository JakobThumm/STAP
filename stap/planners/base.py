import abc
import dataclasses
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from stap import agents, dynamics, envs
from stap.envs.base import Primitive
from stap.envs.pybullet.table_env import Task
from stap.planners.custom_fns import CUSTOM_FNS
from stap.planners.evaluation_fns import EVALUATION_FNS
from stap.utils import tensors


@dataclasses.dataclass
class PlanningResult:
    actions: np.ndarray  # [T, dim_actions]
    states: np.ndarray  # [T+1, dim_states]
    p_success: float
    values: np.ndarray  # [T]
    visited_actions: Optional[np.ndarray] = None  # [num_visited, T, dim_actions]
    visited_states: Optional[np.ndarray] = None  # [num_visited, T+1, dim_states]
    p_visited_success: Optional[np.ndarray] = None  # [num_visited]
    visited_values: Optional[np.ndarray] = None  # [num_visited, T]
    values_unc: Optional[np.ndarray] = None  # [T]
    predicted_preference_values: Optional[np.ndarray] = None  # [T]
    observed_preference_values: Optional[np.ndarray] = None  # [T]


class Planner(abc.ABC):
    """Base planner class."""

    def __init__(
        self,
        policies: Sequence[agents.Agent],
        dynamics: dynamics.Dynamics,
        custom_fns: Optional[
            Dict[Primitive, Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Primitive], torch.Tensor]]]
        ] = None,
        env: Optional[envs.Env] = None,
        device: str = "auto",
    ):
        """Constructs the planner.

        Args:
            policies: Ordered list of policies.
            dynamics: Dynamics model.
            custom_fns: Custom value function to apply at the trajectory evaluation stage (utils.evaluate_trajectory)
            device: Torch device.
        """
        self._policies = policies
        self._dynamics = dynamics
        self._custom_fns = custom_fns
        self._env = env
        self.to(device)

    @property
    def policies(self) -> Sequence[agents.Agent]:
        """Ordered list of policies."""
        return self._policies

    @property
    def dynamics(self) -> dynamics.Dynamics:
        """Dynamics model."""
        return self._dynamics

    @property
    def custom_fns(
        self,
    ) -> Optional[
        Dict[Primitive, Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Primitive], torch.Tensor]]]
    ]:
        """Custom functions."""
        return self._custom_fns

    @property
    def env(self) -> Optional[envs.Env]:
        """Environment."""
        return self._env

    @property
    def device(self) -> torch.device:
        """Torch device."""
        return self._device

    def build_custom_fn_list(self, task: Task, is_custom_fn: bool = True) -> List[Optional[Callable]]:
        """Builds a list of custom functions for each primitive.

        Args:
            task: Task to perform.
            is_custom_fn: Whether to use the custom functions or the evaluation functions.
        Returns:
            List of custom functions.
        """
        # First convert the custom function strings of the task into the actual functions
        task_custom_fns: List[Optional[Callable]] = []
        custom_fns = task.custom_fns if is_custom_fn else task.evaluation_fns
        assert len(custom_fns) == len(task.action_skeleton)
        for fn_name in custom_fns:
            if fn_name is not None:
                if is_custom_fn:
                    assert fn_name in CUSTOM_FNS, f"Custom function {fn_name} not found."
                    task_custom_fns.append(CUSTOM_FNS.get(fn_name))
                else:
                    assert fn_name in EVALUATION_FNS, f"Evaluation function {fn_name} not found."
                    task_custom_fns.append(EVALUATION_FNS.get(fn_name))
            else:
                task_custom_fns.append(None)
        # Now override the task custom functions with the planner custom functions if they exist for a given primitive.
        if self.custom_fns is not None:
            for i in range(len(task.action_skeleton)):
                primitive = task.action_skeleton[i]
                if type(primitive) in self.custom_fns and self.custom_fns.get(type(primitive)) is not None:
                    task_custom_fns[i] = self.custom_fns.get(type(primitive))
        return task_custom_fns

    def to(self, device: Union[str, torch.device]) -> "Planner":
        """Transfers networks to device."""
        self._device = torch.device(tensors.device(device))
        self._dynamics.to(self.device)
        for policy in self.policies:
            policy.to(self.device)
        return self

    @abc.abstractmethod
    def plan(
        self,
        observation: np.ndarray,
        task: Task,
        return_visited_samples: bool = False,
    ) -> PlanningResult:
        """Plans a sequence of actions following the given action skeleton.

        Args:
            observation: Environment observation.
            task: Task to perform.
            return_visited_samples: Whether to return the samples visited during planning.

        Returns:
            Planning result.
        """
        pass
