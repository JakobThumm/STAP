import functools
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from stap import agents, dynamics, envs, networks
from stap.envs.base import Primitive
from stap.envs.pybullet.table.primitives import Pick, StaticHandover
from stap.envs.pybullet.table_env import Task
from stap.planners import base as planners
from stap.planners import utils
from stap.utils import spaces
from stap.utils.tensors import to  # noqa: F401


class CEMPlanner(planners.Planner):
    """Planner using the Improved Cross Entropy Method."""

    def __init__(
        self,
        policies: Sequence[agents.Agent],
        dynamics: dynamics.Dynamics,
        custom_fns: Optional[
            Dict[Primitive, Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Primitive], torch.Tensor]]]
        ] = None,
        env: Optional[envs.Env] = None,
        num_iterations: int = 8,
        num_samples: int = 128,
        num_elites: int = 16,
        standard_deviation: float = 1.0,
        keep_elites_fraction: float = 0.0,
        population_decay: float = 1.0,
        std_decay: float = 0.5,
        device: str = "auto",
    ):
        """Constructs the iCEM planner.

        Args:
            policies: Policies used to evaluate trajecotries.
            dynamics: Dynamics model.
            custom_fns: Custom value function to apply at the trajectory evaluation stage (utils.evaluate_trajectory)
            env: Optional environment to use for planning
            num_iterations: Number of CEM iterations.
            num_samples: Number of samples to generate per CEM iteration.
            num_elites: Number of elites to select from population.
            standard_deviation: Used to sample random actions for the initial
                population. Will be scaled by the action space.
            keep_elites_fraction: Fraction of elites to keep between iterations.
            population_decay: Population decay applied after each iteration.
            std_decay: Decay of standard deviation over time. Std_{t+1} = std_t * (1 - std_decay)
            device: Torch device.
        """
        super().__init__(policies=policies, dynamics=dynamics, custom_fns=custom_fns, env=env, device=device)
        self._num_iterations = num_iterations
        self._num_samples = num_samples
        self._num_elites = max(2, min(num_elites, self.num_samples // 2))
        self._standard_deviation = standard_deviation

        # Improved CEM parameters.
        self._num_elites_to_keep = int(keep_elites_fraction * self.num_elites + 0.5)
        self._population_decay = population_decay
        self._std_decay = std_decay

    @property
    def num_iterations(self) -> int:
        """Number of CEM iterations."""
        return self._num_iterations

    @property
    def num_samples(self) -> int:
        """Number of samples to generate per CEM iteration."""
        return self._num_samples

    @property
    def num_elites(self) -> int:
        """Number of elites to select from population."""
        return self._num_elites

    @property
    def standard_deviation(self) -> float:
        """Unnormalized standard deviation for sampling random actions."""
        return self._standard_deviation

    @property
    def num_elites_to_keep(self) -> int:
        """Number of elites to keep between iterations."""
        return self._num_elites_to_keep

    @property
    def population_decay(self) -> float:
        """Population decay applied after each iteration."""
        return self._population_decay

    @property
    def std_decay(self) -> float:
        """Decay of standard deviation over time."""
        return self._std_decay

    def _compute_initial_distribution(
        self, observation: torch.Tensor, action_skeleton: Sequence[envs.Primitive]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the initial popoulation distribution.

        The mean is generated by randomly rolling out a random trajectory using
        the dynamics model. The standard deviation is scaled according to the
        action space for each action in the skeleton.

        Args:
            observation: Start observation.
            action_skeleton: List of primitives.

        Returns:
            2-tuple (mean, std).
        """
        T = len(action_skeleton)

        # Roll out a trajectory.
        _, actions = self.dynamics.rollout(observation, action_skeleton, self.policies)

        # Scale the standard deviations by the action spaces.
        std = spaces.null_tensor(self.dynamics.action_space, (T,))
        mean = spaces.null_tensor(self.dynamics.action_space, (T,))
        for t, primitive in enumerate(action_skeleton):
            a = self.policies[primitive.idx_policy].action_space
            action_range = torch.from_numpy(a.high - a.low)
            std[t, : action_range.shape[0]] = self.standard_deviation * 0.5 * action_range
            mean[t, : action_range.shape[0]] = actions[t, : action_range.shape[0]]
            if isinstance(primitive, Pick):
                mean[t, :] = torch.zeros_like(mean[t, :], device=self.device)
                std[t, 1] = 0.2 * self.standard_deviation * 0.5 * action_range[1]
                std[t, 2] = 0.2 * self.standard_deviation * 0.5 * action_range[2]
                std[t, 3] = 1.0 * self.standard_deviation * 0.5 * action_range[3]
            elif isinstance(primitive, StaticHandover):
                mean[t, :] = torch.zeros_like(mean[t, :], device=self.device)
                std[t, :] = 2.0 * self.standard_deviation * 0.5 * action_range
        return mean.to(self.device), std.to(self.device)

    def plan(
        self,
        observation: np.ndarray,
        task: Task,
        return_visited_samples: bool = False,
    ) -> planners.PlanningResult:
        """Runs `num_iterations` of CEM.

        Args:
            observation: Environment observation.
            task: Task to plan.
            return_visited_samples: Whether to return the samples visited during planning.

        Returns:
            Planning result.
        """
        action_skeleton = task.action_skeleton
        best_actions: Optional[np.ndarray] = None
        best_states: Optional[np.ndarray] = None
        p_best_success: float = -float("inf")
        best_values: Optional[np.ndarray] = None
        if return_visited_samples:
            visited_actions_list = []
            visited_states_list = []
            p_visited_success_list = []
            visited_values_list = []

        value_fns = [self.policies[primitive.idx_policy].critic for primitive in action_skeleton]
        decode_fns = [functools.partial(self.dynamics.decode, primitive=primitive) for primitive in action_skeleton]
        custom_fns = self.build_custom_fn_list(task)

        with torch.no_grad():
            # Prepare action spaces.
            T = len(action_skeleton)
            actions_low = spaces.null_tensor(self.dynamics.action_space, (T,))
            actions_high = actions_low.clone()
            task_dimensionality = 0
            for t, primitive in enumerate(action_skeleton):
                action_space = self.policies[primitive.idx_policy].action_space
                action_shape = action_space.shape[0]
                actions_low[t, :action_shape] = torch.from_numpy(action_space.low)
                actions_high[t, :action_shape] = torch.from_numpy(action_space.high)
                task_dimensionality += action_shape
            actions_low = actions_low.to(self.device)
            actions_high = actions_high.to(self.device)

            # Scale number of samples to task size
            num_samples = self.num_samples * task_dimensionality

            # Get initial state.
            t_observation = torch.from_numpy(observation).to(self.dynamics.device)

            # Initialize distribution.
            mean, std = self._compute_initial_distribution(t_observation, action_skeleton)
            elites = mean[None, ...]
            elites_scores = torch.ones((1), dtype=torch.float32, device=self.device)
            # from stap.planners.debug import debug_value_fn
            # debug_value_fn(
            #     t_observation, self.policies[action_skeleton[0].idx_policy].action_space, value_fns[0], decode_fns[0]
            # )

            # Prepare constant agents for rollouts.
            policies = [
                agents.ConstantAgent(
                    action=spaces.null_tensor(
                        self.policies[primitive.idx_policy].action_space,
                        num_samples,
                        device=self.device,
                    ),
                    policy=self.policies[primitive.idx_policy],
                )
                for t, primitive in enumerate(action_skeleton)
            ]

            for idx_iter in range(self.num_iterations):
                # Sample from distribution.
                # samples = self.sample_from_nan(mean, std, torch.Size((num_samples,)))
                samples = self.sample_from_elites(elites, elites_scores, std, num_samples)
                num_samples = samples.shape[0]
                samples = torch.clip(samples, actions_low, actions_high)

                # Include the best elites from the previous iteration.
                samples[: elites.shape[0]] = elites

                # Roll out trajectories.
                for t, policy in enumerate(policies):
                    network = policy.actor.network
                    assert isinstance(network, networks.Constant)
                    network.constant = samples[:, t, : policy.action_space.shape[0]]
                states, _ = self.dynamics.rollout(
                    t_observation,
                    action_skeleton,
                    policies,
                    batch_size=num_samples,
                    time_index=True,
                )

                # Evaluate trajectories.
                p_success, values, values_unc = utils.evaluate_trajectory(
                    value_fns=value_fns,
                    decode_fns=decode_fns,
                    states=states,
                    actions=samples,
                    custom_fns=custom_fns,
                    action_skeleton=action_skeleton,
                )
                assert not torch.any(torch.isnan(p_success))
                # Select the top trajectories.
                idx_elites = p_success.topk(self.num_elites).indices
                n_success = torch.sum(p_success[idx_elites] > 0.0)
                if n_success == 0:
                    print("No successful plan found.")
                    n_success = 1
                    # Increase standard deviation
                    std = torch.clip(std * (1 + self.std_decay), 1e-8)
                else:
                    # Reduce standard deviation
                    std = torch.clip(std * (1 - self.std_decay), 1e-8)
                if n_success < idx_elites.shape[0]:
                    idx_elites = idx_elites[:n_success]
                elites = samples[idx_elites]
                elites_scores = p_success[idx_elites]
                idx_best = idx_elites[0]

                # Track best action.
                _p_best_success = p_success[idx_best].cpu().numpy()
                if _p_best_success > p_best_success:
                    p_best_success = _p_best_success
                    best_actions = samples[idx_best].cpu().numpy()
                    best_states = states[idx_best].cpu().numpy()
                    best_values = values[idx_best].cpu().numpy()
                    best_values_unc = values_unc[idx_best].cpu().numpy()

                # Decay population size.
                num_samples = int(self.population_decay * num_samples + 0.5)
                num_samples = max(num_samples, 2 * self.num_elites)

                # Convert to numpy.
                if return_visited_samples:
                    visited_actions_list.append(samples.cpu().numpy())
                    visited_states_list.append(states.cpu().numpy())
                    p_visited_success_list.append(p_success.cpu().numpy())
                    visited_values_list.append(values.cpu().numpy())

        assert (
            best_actions is not None
            and best_states is not None
            and best_values is not None
            and best_values_unc is not None
        )

        if return_visited_samples:
            visited_actions = np.concatenate(visited_actions_list, axis=0)
            visited_states = np.concatenate(visited_states_list, axis=0)
            p_visited_success = np.concatenate(p_visited_success_list, axis=0)
            visited_values = np.concatenate(visited_values_list, axis=0)
        else:
            visited_actions = None
            visited_states = None
            p_visited_success = None
            visited_values = None

        return planners.PlanningResult(
            actions=best_actions,
            states=best_states,
            p_success=p_best_success,
            values=best_values,
            values_unc=best_values_unc,
            visited_actions=visited_actions,
            visited_states=visited_states,
            p_visited_success=p_visited_success,
            visited_values=visited_values,
        )

    def sample_from_nan(self, mean: torch.Tensor, std: torch.Tensor, n_samples: torch.Size):
        """Sample n samples from a torch normal distribution, where some of the entries of mean and std can be nan.
        Replaces nan in mean with 0 and in std with 1.

        Args:
            mean: torch tensor of shape [T, dim_actions]
            std: torch tensor of shape [T, dim_actions]
            n_samples: int
        Returns:
            samples: torch tensor of shape [n_samples, T, dim_actions]
        """
        mean = torch.nan_to_num(mean, nan=0.0)
        std = torch.nan_to_num(std, nan=1.0)
        return torch.distributions.Normal(mean, std).sample(n_samples)

    def sample_from_elites(self, elites: torch.Tensor, elites_scores: torch.Tensor, std: torch.Tensor, n_samples: int):
        """Sample n samples from a torch normal distribution, where some of the entries of mean and std can be nan.
        Replaces nan in mean with 0 and in std with 1.

        Args:
            elites: torch tensor of shape [N, T, dim_actions]
            elites_scores: torch tensor of shape [N, T]
            std: torch tensor of shape [T, dim_actions]
            n_samples: int
        Returns:
            samples: torch tensor of shape [n_samples, T, dim_actions]
        """
        assert elites.shape[0] > 0
        assert elites.shape[0] == elites_scores.shape[0]
        # Figure out how to cast this to int
        sum_elites = torch.sum(elites_scores)
        if sum_elites > 0:
            samples_elites = torch.round(elites_scores / torch.sum(elites_scores) * n_samples).to(torch.int32)
        else:
            samples_elites = (
                torch.ones_like(elites_scores, dtype=torch.int32, device=self.device) * n_samples / elites.shape[0]
            ).to(torch.int32)
        n_samples = torch.sum(samples_elites)  # type: ignore
        # Figure out how to concatenate torch.Size
        samples = torch.zeros([n_samples, *elites[0].shape], device=self.device)
        std = torch.nan_to_num(std, nan=1.0)
        current_idx = 0
        for n_sample_elites, elite in zip(samples_elites, elites):
            mean = torch.nan_to_num(elite, nan=0.0)
            samples[current_idx : current_idx + n_sample_elites] = torch.distributions.Normal(mean, std).sample(
                torch.Size((n_sample_elites,))  # type: ignore
            )
            current_idx += n_sample_elites
        return samples
