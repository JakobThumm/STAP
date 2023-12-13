"""The table environment with an added human in the scene.

Author: Jakob Thumm
Created: 28.11.2023
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p

from stap.envs import base as envs
from stap.envs.pybullet.sim.human import Human
from stap.envs.pybullet.sim.safe_arm import SafeArm
from stap.envs.pybullet.table import object_state, utils
from stap.envs.pybullet.table.primitives import Primitive
from stap.envs.pybullet.table.utils import primitive_from_action_call
from stap.envs.pybullet.table_env import TableEnv
from stap.utils.animation_utils import load_human_animation_data
from stap.utils.macros import SIMULATION_TIME_STEP

dbprint = lambda *args: None  # noqa
# dbprint = print


class HumanTableEnv(TableEnv):
    """The table environment with an added human in the scene."""

    def __init__(
        self,
        human_config: Union[str, Dict[str, Any]],
        animation_type: str,
        animation_frequency: int,
        human_animation_names: Optional[List[str]] = None,
        visualize_shield: bool = False,
        **kwargs,
    ):
        """Constructs the TableEnv.

        Args:
            kwargs: Keyword arguments for `TableEnv`.
        """
        self._human_kwargs = utils.load_config(human_config)
        self._animations = load_human_animation_data(
            animation_type=animation_type,
            human_animation_names=human_animation_names,
            load_post_processed=True,
            verbose=False,
        )
        self._visualize_shield = visualize_shield
        super().__init__(**kwargs)
        self.human.reset(self.task.action_skeleton, self.task.initial_state)
        self._initial_state_id = p.saveState(physicsClientId=self.physics_id)
        # Animation: Time, Point, Pos
        self._animation_freq = animation_frequency
        self._sim_time = 0.0
        self._animation_number = 0
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=90,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0],
            physicsClientId=self.physics_id,
        )

    def _create_objects(self, object_kwargs: Optional[List[Dict[str, Any]]] = None) -> None:
        super()._create_objects(object_kwargs=object_kwargs)
        self.human = Human(self.physics_id, **self._human_kwargs)
        human_hands = self.human.get_hand_objects()
        self._objects.update(human_hands)

    def reset(  # type: ignore
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        max_samples_per_trial: int = 100,
    ) -> Tuple[np.ndarray, dict]:
        self.human.reset(self.task.action_skeleton, self.task.initial_state)
        self._animation_number += 1
        if self._animation_number >= len(self._animations):
            self._animation_number = 0
        animation, animation_info = self._animations[self._animation_number]
        self.human.set_animation(animation, animation_info, self._animation_freq)
        if self._sim_time is None:
            self._sim_time = 0.0
        self.human.disable_animation()
        self._sim_time = 0.0
        obs, info = super().reset(seed=seed, options=options, max_samples_per_trial=max_samples_per_trial)
        self.human.enable_animation()
        self.human.animate(self._sim_time)
        return obs, info

    def set_primitive(
        self,
        primitive: Optional[envs.Primitive] = None,
        action_call: Optional[str] = None,
        idx_policy: Optional[int] = None,
        policy_args: Optional[Any] = None,
    ) -> envs.Env:
        if primitive is None:
            primitive = self.get_primitive_info(action_call=action_call, idx_policy=idx_policy, policy_args=policy_args)
        assert isinstance(primitive, Primitive)
        self._primitive = primitive

        return self

    def get_primitive_info(
        self,
        action_call: Optional[str] = None,
        idx_policy: Optional[int] = None,
        policy_args: Optional[Any] = None,
    ) -> envs.Primitive:
        if action_call is not None:
            return primitive_from_action_call(action_call, self)
        elif idx_policy is not None and policy_args is not None:
            arg_indices = [
                idx_obs - 1 if idx_obs > TableEnv.EE_OBSERVATION_IDX else idx_obs
                for idx_obs in policy_args["observation_indices"][: policy_args["shuffle_range"][0]]
                if idx_obs != TableEnv.EE_OBSERVATION_IDX
            ]
            object_names = list(self.objects.keys())
            args = ", ".join(object_names[idx_obj] for idx_obj in arg_indices)
            action_call = f"{self.primitives[idx_policy]}({args})"
            return primitive_from_action_call(action_call, self)
        else:
            raise ValueError("One of action_call or (idx_policy, policy_args) must not be None.")

    def get_state(self) -> np.ndarray:
        return super().get_state()

    def set_state(self, state: np.ndarray) -> bool:
        return super().set_state(state)

    def get_observation(self, image: Optional[bool] = None) -> np.ndarray:
        """Gets the current low-dimensional state for all the objects.

        The observation is a [MAX_NUM_OBJECTS, d] matrix, where d is the length
        of the low-dimensional object state. The first row corresponds to the
        pose of the end-effector, and the following rows correspond to the
        states of all the objects in order. Any remaining rows are zero.
        """
        return super().get_observation(image=image)

    def set_observation(self, observation: np.ndarray) -> None:
        """Sets the object states from the given low-dimensional state observation.

        See `TableEnv.get_observation()` for a description of the observation.
        """
        super().set_observation(observation)

    def object_states(self) -> Dict[str, object_state.ObjectState]:
        """Returns the object states as an ordered dict indexed by object name.

        The first item in the dict corresponds to the end-effector pose.
        """
        return super().object_states()

    def step(
        self, action: np.ndarray, custom_recording_text: Optional[Dict[str, str]] = None
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated_result, info = super().step(
            action, custom_recording_text=custom_recording_text
        )
        return obs, reward, terminated, truncated_result, info

    def step_simulation(self) -> float:
        self._sim_time += SIMULATION_TIME_STEP
        self.human.animate(self._sim_time)
        super().step_simulation()
        if self.robot._arm_class == SafeArm:
            self.robot.arm.human_measurement(self._sim_time, self.human.measurement(self._sim_time))
            if self.gui and self._visualize_shield:
                init_before = self.robot.arm.visualization_initialized
                self.robot.arm.visualize()
                if not init_before and self.robot.arm.visualization_initialized:
                    # We have to save the state after the first visualization, to include the shield markers.
                    self._initial_state_id = p.saveState(physicsClientId=self.physics_id)
        human_contact_points = p.getContactPoints(
            physicsClientId=self.physics_id, bodyA=self.robot.arm.body_id, bodyB=self.human.body_id
        )
        if len(human_contact_points) > 0:
            stop = True
        return self._sim_time

    def render(self) -> np.ndarray:  # type: ignore
        img = super().render()
        return img
