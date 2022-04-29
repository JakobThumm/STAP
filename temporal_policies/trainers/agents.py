import pathlib
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np  # type: ignore
import torch  # type: ignore
import tqdm  # type: ignore

from temporal_policies import agents, datasets, envs, processors
from temporal_policies.schedulers import DummyScheduler
from temporal_policies.trainers.base import Trainer
from temporal_policies.utils import configs, metrics, tensors
from temporal_policies.utils.typing import Batch


class AgentTrainer(Trainer[agents.RLAgent, Batch, Batch]):
    """Agent trainer."""

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        agent: agents.RLAgent,
        dataset_class: Union[
            str, Type[torch.utils.data.IterableDataset]
        ] = datasets.ReplayBuffer,
        dataset_kwargs: Dict[str, Any] = {},
        eval_dataset_kwargs: Optional[Dict[str, Any]] = None,
        processor_class: Union[
            str, Type[processors.Processor]
        ] = processors.IdentityProcessor,
        processor_kwargs: Dict[str, Any] = {},
        optimizer_class: Union[str, Type[torch.optim.Optimizer]] = torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3},
        scheduler_class: Union[
            str, Type[torch.optim.lr_scheduler._LRScheduler]
        ] = DummyScheduler,
        scheduler_kwargs: Dict[str, Any] = {},
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
        num_pretrain_steps: int = 1000,
        num_train_steps: int = 100000,
        num_eval_episodes: int = 100,
        eval_freq: int = 1000,
        checkpoint_freq: int = 10000,
        log_freq: int = 100,
        profile_freq: Optional[int] = None,
        eval_metric: str = "reward",
        num_data_workers: int = 0,
    ):
        """Prepares the agent trainer for training.

        Args:
            path: Training output path.
            agent: Agent to be trained.
            dataset_class: Dynamics model dataset class or class name.
            dataset_kwargs: Kwargs for dataset class.
            eval_dataset_kwargs: Kwargs for eval dataset.
            processor_class: Batch data processor calss.
            processor_kwargs: Kwargs for processor.
            optimizer_class: Dynamics model optimizer class.
            optimizer_kwargs: Kwargs for optimizer class.
            scheduler_class: Optional optimizer scheduler class.
            scheduler_kwargs: Kwargs for scheduler class.
            checkpoint: Optional path to trainer checkpoint.
            device: Torch device.
            num_pretrain_steps: Number of steps to pretrain.
            num_train_steps: Number of steps to train.
            num_eval_episodes: Number of episodes per evaluation.
            eval_freq: Evaluation frequency.
            checkpoint_freq: Checkpoint frequency (separate from latest/best
                eval checkpoints).
            log_freq: Logging frequency.
            profile_freq: Profiling frequency.
            eval_metric: Metric to use for evaluation.
            num_data_workers: Number of workers to use for dataloader.
        """
        path = pathlib.Path(path) / agent.env.name

        dataset_class = configs.get_class(dataset_class, datasets)
        dataset_kwargs = dict(dataset_kwargs)
        dataset_kwargs["path"] = path / "train_data"
        dataset = dataset_class(
            observation_space=agent.observation_space,
            action_space=agent.action_space,
            **dataset_kwargs,
        )

        if eval_dataset_kwargs is None:
            eval_dataset_kwargs = dataset_kwargs
        eval_dataset_kwargs = dict(eval_dataset_kwargs)
        eval_dataset_kwargs["save_frequency"] = None
        eval_dataset_kwargs["path"] = path / "eval_data"
        eval_dataset = dataset_class(
            observation_space=agent.observation_space,
            action_space=agent.action_space,
            **eval_dataset_kwargs,
        )

        processor_class = configs.get_class(processor_class, processors)
        processor = processor_class(
            agent.observation_space, agent.action_space, **processor_kwargs
        )

        optimizer_class = configs.get_class(optimizer_class, torch.optim)
        optimizers = agent.create_optimizers(optimizer_class, optimizer_kwargs)

        scheduler_class = configs.get_class(scheduler_class, torch.optim.lr_scheduler)
        schedulers = {
            key: scheduler_class(optimizer, **scheduler_kwargs)
            for key, optimizer in optimizers.items()
        }

        super().__init__(
            path=path,
            model=agent,
            dataset=dataset,
            eval_dataset=eval_dataset,
            processor=processor,
            optimizers=optimizers,
            schedulers=schedulers,
            checkpoint=checkpoint,
            device=device,
            num_pretrain_steps=num_pretrain_steps,
            num_train_steps=num_train_steps,
            num_eval_steps=num_eval_episodes,
            eval_freq=eval_freq,
            checkpoint_freq=checkpoint_freq,
            log_freq=log_freq,
            profile_freq=profile_freq,
            eval_metric=eval_metric,
            num_data_workers=num_data_workers,
        )

    @property
    def agent(self) -> agents.RLAgent:
        """Agent being trained."""
        return self.model

    @property
    def env(self) -> envs.Env:
        """Agent env."""
        return self.agent.env

    def collect_step(self, random: bool = False) -> Dict[str, Any]:
        """Collects data for the replay buffer.

        Args:
            random: Use random actions.

        Returns:
            Collect metrics.
        """
        with self.profiler.profile("collect"):
            if self.step == 0:
                self.dataset.add(observation=self.env.reset())
                self._episode_length = 0
                self._episode_reward = 0

            if random:
                action = self.agent.action_space.sample()
            else:
                self.eval_mode()
                with torch.no_grad():
                    observation = tensors.from_numpy(
                        self.env.get_observation(), self.device
                    )
                    action = self.agent.actor.predict(self.agent.encoder(observation))
                    action = action.cpu().numpy()
                self.train_mode()

            next_observation, reward, done, info = self.env.step(action)
            discount = 1.0 - done

            self.dataset.add(
                action=action,
                reward=reward,
                next_observation=next_observation,
                discount=discount,
                done=done,
            )

            self._episode_length += 1
            self._episode_reward += reward
            if not done:
                return {}

            self.increment_epoch()

            metrics = {
                "reward": self._episode_reward,
                "length": self._episode_length,
                "episode": self.epoch,
            }

            # Reset the environment
            self.dataset.add(observation=self.env.reset())
            self._episode_length = 0
            self._episode_reward = 0

            return metrics

    def pretrain(self) -> None:
        """Runs the pretrain phase."""
        pbar = tqdm.tqdm(
            range(self.step, self.num_pretrain_steps),
            desc=f"Pretrain {self.name}",
            dynamic_ncols=True,
        )
        metrics_list = []
        for _ in pbar:
            collect_metrics = self.collect_step(random=True)
            pbar.set_postfix({self.eval_metric: collect_metrics[self.eval_metric]})

            metrics_list.append(collect_metrics)
            metrics_list = self.log_step(metrics_list, stage="pretrain")

            self.increment_step()

    def train_step(self, step: int, batch: Batch) -> Dict[str, float]:
        """Performs a single training step.

        Args:
            step: Training step.
            batch: Training batch.

        Returns:
            Dict of training metrics for logging.
        """
        collect_metrics = self.collect_step(random=False)
        train_metrics = super().train_step(step, batch)

        return {**collect_metrics, **train_metrics}

    def evaluate(self) -> List[Dict[str, np.ndarray]]:
        """Evaluates the model.

        Returns:
            Eval metrics.
        """
        self.eval_mode()

        with self.profiler.profile("evaluate"):
            metrics_list = []
            pbar = tqdm.tqdm(
                range(self.num_eval_steps),
                desc=f"Eval {self.name}",
                dynamic_ncols=True,
            )
            for _ in pbar:
                observation = self.env.reset()
                self.eval_dataset.add(observation=observation)

                step_metrics_list = []
                done = False
                while not done:
                    with torch.no_grad():
                        observation = tensors.from_numpy(observation, self.device)
                        action = self.agent.actor.predict(
                            self.agent.encoder(observation)
                        )
                        action = action.cpu().numpy()

                    observation, reward, done, info = self.env.step(action)
                    self.eval_dataset.add(
                        action=action,
                        reward=reward,
                        next_observation=observation,
                        discount=1.0 - done,
                        done=done,
                    )

                    step_metrics = {
                        metric: value
                        for metric, value in info.items()
                        if metric in metrics.METRIC_AGGREGATION_FNS
                    }
                    step_metrics["reward"] = reward
                    step_metrics["length"] = 1
                    step_metrics_list.append(step_metrics)

                episode_metrics = metrics.aggregate_metrics(step_metrics_list)
                metrics_list.append(episode_metrics)
                pbar.set_postfix({self.eval_metric: episode_metrics[self.eval_metric]})

            self.eval_dataset.save()

        self.train_mode()

        return metrics_list
