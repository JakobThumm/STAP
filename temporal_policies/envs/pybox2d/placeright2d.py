import numpy as np
from gym import spaces
from Box2D import *

from .base import Box2DBase
from .utils import shape_to_vertices


class PlaceRight2D(Box2DBase):

    def __init__(self, **kwargs):
        """PlaceRight2D gym environment.
        """
        super().__init__(**kwargs)
        self.agent = None
        self.reset()
        
    def reset(self):
        observation = super().reset()
        return observation

    def step(self, action):
        """Action components are activated via tanh().
        """
        # Act
        action = action.astype(float)
        low, high = self.action_scale.low, self.action_scale.high
        action = low + (high - low) * ((action + 1) / 2)
        self.agent.position = b2Vec2(action[0], self.agent.position[1])
        self.agent.angle = action[1]
        self.agent.fixedRotation = True

        # Simulate
        steps_exceeded = super().step() 
        observation = self._get_observation()
        reward = self._get_reward(observation)
        done = steps_exceeded or self._get_done()
        info = {}
        return observation, reward, done, info
    
    def _setup_spaces(self):
        """PlaceRight2D primitive action and observation spaces.
        Action space: (self.agent.position.x, self.agent.position.angle)
        Observation space: [Bounding box parameters of all 2D rigid bodies]
        """
        # Agent
        self.agent = self._get_body("item", "block")

        # Space params
        item_w = max(self._get_shape_kwargs("item")["size"])
        wksp_pos_x, wksp_pos_y = self._get_shape("playground", "ground")["position"]
        wksp_w, wksp_h = self._get_shape_kwargs("playground")["size"]
        wksp_t = self._get_shape_kwargs("playground")["t"]
        
        # Action space
        x_min = wksp_pos_x - wksp_w / 2 + item_w / 2
        x_max = wksp_pos_x + wksp_w / 2 - item_w / 2
        self.action_scale = spaces.Box(
            low=np.array([x_min, -np.pi/2], dtype=np.float32),
            high=np.array([x_max, np.pi/2], dtype=np.float32)
        )
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32)
        )
        
        # Observation space
        x_min = wksp_pos_x - wksp_w / 2 - wksp_t
        x_max = wksp_pos_x + wksp_w / 2 + wksp_t
        y_min = wksp_pos_y - wksp_t / 2
        y_max = wksp_pos_y + wksp_t / 2 + wksp_h
        w_min, w_max = wksp_t / 2, wksp_w / 2 + wksp_t
        h_min, h_max = wksp_t / 2, wksp_h / 2

        all_bodies = set([body.userData for body in self.world.bodies])
        redundant_bodies = set([*self.env["playground"]["bodies"].keys(), self.agent.userData])
        self._observation_bodies = all_bodies - redundant_bodies

        reps = len(self._observation_bodies)
        self.observation_space = spaces.Box(
            low=np.tile(np.array([x_min, y_min, w_min, h_min], dtype=np.float32), reps), 
            high=np.tile(np.array([x_max, y_max, w_max, h_max], dtype=np.float32), reps)
        )
        
    def _get_observation(self):
        k = 0
        observation = np.zeros((self.observation_space.shape[0]), dtype=np.float32)
        for _, object_data in self.env.items():
            for shape_name, shape_data in object_data["shapes"].items():
                if shape_name not in self._observation_bodies: continue
                position = np.array(object_data["bodies"][shape_name].position, dtype=np.float32)
                observation[k: k+4] = np.concatenate((position, shape_data["box"]))
                k += 4
        assert self.observation_space.contains(observation)
        return observation

    def _get_reward(self, observation):
        """PlaceRight2D reward function.
            - reward=1.0 iff block touches ground and to the right of receptacle box
            - reward=0.0 otherwise
        """
        on_ground = False
        for contact in self.agent.contacts:
            if contact.other.userData == self._get_body_name("playground", "ground"):
                on_ground = True
                break
        
        box_vertices = shape_to_vertices(
            position=self._get_body("box", "ceiling").position,
            box=self._get_shape("box", "ceiling")["box"]
        )
        x_min = np.amax(box_vertices, axis=0)[0]
        on_right = self.agent.position[0] >= x_min
        reward = float(on_ground and on_right)
        return reward

    def _get_done(self):
        return True
    