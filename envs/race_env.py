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

        self.track_name = "track"
        self.runoff_name = "runoff"

        self.max_steps = 10_000
        self.step_count = 0

        self.max_torque = 0
        self.max_brake = 0
        self.max_steer = 1.0

    def _setup_env(self):

        current_path = Path(__file__).resolve()
        circuit_data_path = current_path.parent.parent / "circuitData"

        track_file_path = circuit_data_path / self.track_name / ".obj"
        runoff_file_path = circuit_data_path / self.runoff_name / ".obj"

        self.close()
        self.engine_id = p.connect(
            p.GUI if self.render else p.DIRECT
        )

        p.setAdditionalSearchPath(pd.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # load track and car
        base_pos = [0, 0, 0]
        base_orient = p.getQuaternionFromEuler([0, 0, 0])

        track_vis_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=track_file_path,
            meshScale=[1.0, 1.0, 1.0],
            rgbaColor=[0.5, 0.5, 0.5, 1],
            specularColor=[0, 0, 0]
        )

        track_coll_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=track_file_path,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            meshScale=[1.0, 1.0, 1.0]
        )

        track_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=track_coll_id,
            baseVisualShapeIndex=track_vis_id,
            basePosition=base_pos,
            baseOrientation=base_orient
        )

        runoff_vis_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=runoff_file_path,
            meshScale=[1.0, 1.0, 1.0],
            rgbaColor=[0.2, 0.45, 0.2, 1],
            specularColor=[0, 0, 0]
        )

        runoff_coll_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=runoff_file_path,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            meshScale=[1.0, 1.0, 1.0]
        )

        runoff_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=runoff_coll_id,
            baseVisualShapeIndex=runoff_vis_id,
            basePosition=base_pos,
            baseOrientation=base_orient
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        obs = np.zeros(self.observation_space.shape)
        return obs, {}

    def close(self):
        if self.engine_id is not None:
            p.disconnect()
            self.engine_id = None
