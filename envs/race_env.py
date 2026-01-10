import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data as pd
from gymnasium import spaces

from envs.car import Car
from main import config

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent


class RacingEnv(gym.Env):
    def __init__(self,  car_pos, car_orn, render=False):
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

        self.track_name = config.CIRCUIT["track"]
        self.runoff_name = config.CIRCUIT["runoff"]

        # *============ params ============
        self.max_steps = config.MAX_STEPS
        self.step_count = 0
        self.off_ground_count = 0

        self.max_torque = config.CAR["max_torque"]
        self.max_brake_force = config.CAR["max_brake_force"]
        self.max_steer_angle = config.CAR["max_steer_angle"]

        self.lap_started = False
        self.start_time = 0.0
        self.prev_inside = False
        # *================================

        self._setup_env(car_pos, car_orn)

    def _setup_env(self, car_pos, car_orn):
        circuit_data_path = PROJECT_ROOT / config.CIRCUIT["path"]

        track_file_path = str(circuit_data_path / (self.track_name + ".obj"))
        runoff_file_path = str(circuit_data_path / (self.runoff_name + ".obj"))

        self.close()
        self.engine_id = p.connect(
            p.GUI if self.render else p.DIRECT,
            options="--stderr_logging_level=0"
        )

        p.setAdditionalSearchPath(pd.getDataPath())
        p.resetSimulation()
        p.setGravity(*config.GRAVITY)
        p.setTimeStep(1. / 240.)

        # load track and car
        track_base_pos = [0, 0, 0]
        track_base_orient = p.getQuaternionFromEuler([0, 0, 0])

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

        self.track_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=track_coll_id,
            baseVisualShapeIndex=track_vis_id,
            basePosition=track_base_pos,
            baseOrientation=track_base_orient
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

        self.runoff_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=runoff_coll_id,
            baseVisualShapeIndex=runoff_vis_id,
            basePosition=track_base_pos,
            baseOrientation=track_base_orient
        )

        goal_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[config.CIRCUIT["width"]/2, 0.05, 0.01],
            rgbaColor=[1, 0, 0, 0.4]
        )

        self.goal_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=goal_vis,
            basePosition=[config.CIRCUIT["radius"], 0, 0.02]
        )

        self.car = Car(car_pos, car_orn)

        p.changeDynamics(
            self.track_id,
            -1,
            lateralFriction=config.FRICTION["track"]
        )
        p.changeDynamics(
            self.runoff_id,
            -1,
            lateralFriction=config.FRICTION["runoff"]
        )

        for i in range(50):
            p.stepSimulation()

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.car.car_id)
        vel, ang_vel = p.getBaseVelocity(self.car.car_id)

        sensor = self.car.checkHit()
        sensor_flat = np.concatenate(sensor)

        obs = np.concatenate([
             np.array(pos),
             np.array(vel),
             sensor_flat
        ])

        return obs.astype(np.float32)

    def _accross_goal(self):
        pos, orn = p.getBasePositionAndOrientation(self.car.car_id)
        p_car = np.array(pos[:2])

        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        d = rot[:, 1]
        d2 = d[:2]
        d2 = d2 / np.linalg.norm(d2)
        n = np.array([-d2[1], d2[0]])

        rel = p_car - np.array([config.CIRCUIT["radius"], 0])
        long = np.dot(rel, d2)
        lat = np.dot(rel, n)

        inside = (
            0.0 <= long <= 0.1 and
            abs(lat) <= config.CIRCUIT["width"] / 2
        )

        return inside

    def _calc_reward(self, obs):
        vel = obs[3:6]
        reward = np.linalg.norm(vel) * 10

        return reward

    def _check_terminate(self, obs):
        return False

    def _update_cam_pos(self):
        pos, orn = p.getBasePositionAndOrientation(self.car.car_id)
        yaw = p.getEulerFromQuaternion(orn)[2]

        p.resetDebugVisualizerCamera(
            cameraDistance=4.0,
            cameraYaw=np.degrees(yaw),
            cameraPitch=-45,
            cameraTargetPosition=pos
        )

    def reset(self, seed=None, options=None):
        init_pos = [
            config.CAR["base_x"],
            config.CAR["base_y"],
            config.CAR["base_z"]
        ]
        init_orn = p.getQuaternionFromEuler([0, 0, 0])

        super().reset(seed=seed)

        self.car.reset(init_pos, init_orn)
        self.step_count = 0
        self.off_ground_count = 0

        self.prev_inside = False
        self.start_time = 0.0

        obs = self._get_obs()

        for i in range(50):
            p.stepSimulation()

        return obs, {}

    def step(self, action):
        self.step_count += 1

        throttle, brake, steer = action

        self.car.apply_action(
            throttle=throttle,
            brake=brake,
            steer=steer,
            max_torque=self.max_torque,
            max_brake=self.max_brake_force,
            max_steer=self.max_steer_angle
        )

        p.stepSimulation()

        inside = self._accross_goal()
        if inside and not self.prev_inside:
            self.prev_inside = True

            if not self.lap_started:
                self.lap_started = True
                self.start_time = time.time()

            else:
                lap_time = time.time() - self.start_time
                print(f"Lap: {lap_time:.2f}s")
                # self.lap_started = False

        elif not inside and self.prev_inside:
            self.prev_inside = False

        obs = self._get_obs()
        reward = self._calc_reward(obs)

        off_all_wheels = self.car.is_all_wheels_off(self.track_id)
        self.off_ground_count += 1 if off_all_wheels else 0

        terminated = self._check_terminate(obs) or (self.off_ground_count > 3)
        truncated = self.step_count >= self.max_steps

        if terminated:
            reward -= 10.0

        if self.render:
            if not self.car.is_all_wheels_off(self.track_id):
                self._update_cam_pos()
                self.car.draw_car_info(throttle, brake, steer)

        return obs, reward, terminated, truncated, {}

    def close(self):
        if self.engine_id is not None:
            p.disconnect()
            self.engine_id = None
