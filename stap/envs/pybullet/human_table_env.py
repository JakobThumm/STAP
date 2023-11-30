"""The table environment with an added human in the scene.

Author: Jakob Thumm
Created: 28.11.2023
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p

from stap.envs import base as envs
from stap.envs.pybullet.base import SIM_TIME_STEP
from stap.envs.pybullet.sim import math
from stap.envs.pybullet.sim.human import Human
from stap.envs.pybullet.table import object_state, utils
from stap.envs.pybullet.table.primitives import Primitive
from stap.envs.pybullet.table_env import CameraView, TableEnv
from stap.utils.animation_utils import load_human_animation_data

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
        **kwargs,
    ):
        """Constructs the TableEnv.

        Args:
            kwargs: Keyword arguments for `TableEnv`.
        """
        super().__init__(**kwargs)
        human_kwargs = utils.load_config(human_config)
        self._animations = load_human_animation_data(
            animation_type=animation_type,
            human_animation_names=human_animation_names,
            load_post_processed=True,
            verbose=False,
        )
        self.human = Human(self.physics_id, **human_kwargs)
        self.human.reset(self.task.action_skeleton, self.task.initial_state)
        self._initial_state_id = p.saveState(physicsClientId=self.physics_id)
        # Animation: Time, Point, Pos
        self._animation_freq = animation_frequency
        self._sim_time = 0.0
        self._animation_number = 0
        WIDTH, HEIGHT = 405, 405
        PROJECTION_MATRIX = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.5,
            nearVal=0.02,
            farVal=100,
        )
        gui_kwargs = kwargs["gui_kwargs"] if "gui_kwargs" in kwargs else {}
        shadows = gui_kwargs.get("shadows", 0)
        self._camera_views["front"] = CameraView(
            width=WIDTH,
            height=HEIGHT,
            view_matrix=p.computeViewMatrix(
                cameraEyePosition=[3.0, 0.0, 1.0],
                cameraTargetPosition=[0.0, 0.0, 0.1],
                cameraUpVector=[0.0, 0.0, 1.0],
            ),
            projection_matrix=PROJECTION_MATRIX,
            shadow=shadows,
        )
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=90,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0],
            physicsClientId=self.physics_id,
        )

    def reset(  # type: ignore
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        max_samples_per_trial: int = 100,
    ) -> Tuple[np.ndarray, dict]:
        self.human.disable_animation()
        obs, info = super().reset(seed=seed, options=options, max_samples_per_trial=max_samples_per_trial)
        self.human.reset(self.task.action_skeleton, self.task.initial_state)
        self._animation_number += 1
        if self._animation_number >= len(self._animations):
            self._animation_number = 0
        animation, animation_info = self._animations[self._animation_number]
        self.human.set_animation(animation, animation_info, self._animation_freq)
        self.human.enable_animation()
        self._sim_time = 0.0
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
            return Primitive.from_action_call(action_call, self)
        elif idx_policy is not None and policy_args is not None:
            arg_indices = [
                idx_obs - 1 if idx_obs > TableEnv.EE_OBSERVATION_IDX else idx_obs
                for idx_obs in policy_args["observation_indices"][: policy_args["shuffle_range"][0]]
                if idx_obs != TableEnv.EE_OBSERVATION_IDX
            ]
            object_names = list(self.objects.keys())
            args = ", ".join(object_names[idx_obj] for idx_obj in arg_indices)
            action_call = f"{self.primitives[idx_policy]}({args})"
            return Primitive.from_action_call(action_call, self)
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

    def step_simulation(self) -> None:
        self._sim_time += SIM_TIME_STEP
        self.human.animate(self._sim_time)
        super().step_simulation()
        human_contact_points = p.getContactPoints(
            physicsClientId=self.physics_id, bodyA=self.robot.arm.body_id, bodyB=self.human.body_id
        )
        if len(human_contact_points) > 0:
            stop = True

    def wait_until_stable(self, min_iters: int = 0, max_iters: int = 3 * math.PYBULLET_STEPS_PER_SEC) -> int:
        IS_MOVING_KEY = "TableEnv.wait_until_stable"

        def is_any_object_moving() -> bool:
            return any(
                not obj.is_static and utils.is_moving(obj, use_history=IS_MOVING_KEY) for obj in self.real_objects()
            )

        # Reset history for `utils.is_moving()`.
        utils.TWIST_HISTORY[IS_MOVING_KEY].clear()

        num_iters = 0
        while (
            num_iters == 0  # Need to step at least once to update collisions.
            or num_iters < max_iters
            and (num_iters < min_iters or is_any_object_moving())
            and not self._is_any_object_below_table()
            and not self._is_any_object_touching_base()
            and not self._is_any_object_falling_off_parent()
        ):
            self.robot.arm.update_torques()
            self.robot.gripper.update_torques()
            self.step_simulation()
            num_iters += 1

        # print("TableEnv.wait_until_stable: {num_iters}")
        return num_iters

    def render(self) -> np.ndarray:  # type: ignore
        img = super().render()
        return img
