import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data as pd
from gymnasium import spaces
from pathlib import Path


class RacingEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()

        self.render = render
        self.engine_id = None

        # Pos(x, y, z), Vel(vx, vy, vz), Sensor(18),
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24, ),
            dtype=np.float32
        )

        # Throttle, Brake, Steer
        self.action_space = spaces.Box(
            low=np.array([0, 0, -1]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )

        self.max_steps = 10_000
        self.step_count = 0

        self.max_torque = 0
        self.max_brake = 0
        self.max_steer = 1.0

    def _setup_env(self):
        current_path = Path(__file__).

        self.close()
        self.engine_id = p.connect(
            p.GUI if self.render else p.DIRECT
        )

        p.setAdditionalSearchPath(pd.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # load track and car
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        obs = np.zeros(self.observation_space.shape)
        return obs, {}

    def close(self):
        if self.engine_id is not None:
            p.disconnect()
            self.engine_id = None
