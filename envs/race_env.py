import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data as pd
from gymnasium import spaces

from assets.trackMaker.track_info_generator import (gen_center_point,
                                                    gen_mesh_data)
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
            shape=(21, ),
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
        self.time_step = 1. / 240.
        self.sim_time = 0.0
        self.max_time = config.MAX_TIME

        self.step_count = 0
        self.off_ground_count = 0
        self.lap_count = 0
        self.total_lap_count = 0

        self.max_torque = config.CAR["max_torque"]
        self.max_brake_force = config.CAR["max_brake_force"]
        self.max_steer_angle = config.CAR["max_steer_angle"]

        self.lap_started = False
        self.start_time = time.time()
        self.goal_prev_inside = False
        self.left_start = False
        # *================================

        self._setup_env(car_pos, car_orn)

        self.env_id = id(self)

    def _setup_env(self, car_pos, car_orn):
        circuit_data_path = PROJECT_ROOT / config.CIRCUIT["path"]

        track_file_path = str(circuit_data_path / (self.track_name + ".obj"))
        runoff_file_path = str(circuit_data_path / (self.runoff_name + ".obj"))

        self.center_point = gen_center_point(
            config.CIRCUIT["straight"],
            config.CIRCUIT["radius"]
        )
        _, self.track_normal_vec = gen_mesh_data(
            self.center_point,
            config.CIRCUIT["width"],
            config.CIRCUIT["radius"],
            in_out="in"
        )

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

        self.obj_dict = {
            "track": self.track_id,
            "runoff": self.runoff_id
        }

        for i in range(50):
            p.stepSimulation()
            self._update_cam_pos()

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.car.car_id)
        vel, ang_vel = p.getBaseVelocity(self.car.car_id)

        sensor = self.car.checkHit(self.obj_dict)
        sensor_flat = np.concatenate(sensor)

        obs = np.concatenate([
             np.array(vel),
             sensor_flat
        ])

        return obs.astype(np.float32), sensor_flat, pos

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

    def get_nn_index(self, pos):
        points = self.center_point
        car_pos = np.array(pos[:2]) + np.array([0, 0.5])

        diff = points - car_pos
        dists = np.linalg.norm(diff, axis=1)

        idx = int(np.argmin(dists))
        idx_next = (idx - 10) % len(points)

        return idx, idx_next

    def get_runoff_ratio(self, sensor):
        forward = sensor[:7]
        side = sensor[7:13]
        back = sensor[13:18]

        count_foward = 0.0
        count_side = 0.0
        count_back = 0.0

        for i in forward:
            if i == -1:
                count_foward += 1
        for j in side:
            if j == -1:
                count_side += 1
        for k in back:
            if k == -1:
                count_back += 1

        ratio_forward = count_foward / len(forward)
        ratio_side = count_foward / len(side)
        ratio_back = count_back / len(back)

        return ratio_forward, ratio_side, ratio_back

    def _calc_reward(self, obs, pos, steer, sensor):
        reward = 0.0
        nn_idx, nn_indx_next = self.get_nn_index(pos)
        idx_point = self.track_normal_vec[nn_idx]
        idx_point_next = self.track_normal_vec[nn_indx_next]

        car_vel = np.array(obs[:2])
        speed = np.linalg.norm(car_vel)

        idle_penalty = 0.0
        if speed < 1.0:
            idle_penalty = -0.5
            car_vel_unit = np.zeros_like(car_vel)
        else:
            car_vel_unit = car_vel / speed
        tangent_vec = np.array(idx_point) * -1
        tangent_vec_next = np.array(idx_point_next) * -1
        dot_near = np.dot(tangent_vec, car_vel_unit)
        dot_far = np.dot(tangent_vec_next, car_vel_unit)

        forward_speed = np.dot(tangent_vec, car_vel)

        dot_speed_penalty = 0.0
        dir_reward = 0.0
        forward_speed_reward = 0.0
        if dot_near <= 0 or forward_speed <= 0:
            dot_speed_penalty = dot_near * 5.0
        else:
            dir_reward += dot_near
            forward_speed_reward = max(0.0, forward_speed) * 2.0

        wheel_contact_penalty = 0.0
        wheel_contacts = self.car.get_wheel_contact(self.track_id)
        for c in wheel_contacts:
            if not c:
                wheel_contact_penalty -= 3.0

        sensor_penalty = 0.0
        r_f, r_s, r_b = self.get_runoff_ratio(sensor)
        fusion_sensor = r_f * 3.0 + r_s * 1.5 + r_b * 1.0
        sensor_penalty = - fusion_sensor * 1.5

        lookahead = 1 - (dot_near - dot_far)
        lookahead = np.clip(lookahead, 0.0, 1.0)
        steer_penalty = -abs(steer) * lookahead * 1.0

        reward = (
            idle_penalty
            + dot_speed_penalty
            + dir_reward
            + forward_speed_reward
            + wheel_contact_penalty
            + sensor_penalty
            + steer_penalty
        )

        if self.render:
            point = np.append(self.center_point[nn_idx], 0.2)
            point2 = np.append(self.center_point[nn_indx_next], 0.2)

            p.addUserDebugLine(
                point,
                point + np.append(tangent_vec * 0.4, 0.0),
                [1, 0, 0],
                lineWidth=5,
                lifeTime=0.05
            )

            p.addUserDebugLine(
                point2,
                point2 + np.append(tangent_vec_next * 0.4, 0.0),
                [0, 0, 1],
                lineWidth=5,
                lifeTime=0.05
            )

        return reward

    def _update_cam_pos(self):
        pos, orn = p.getBasePositionAndOrientation(self.car.car_id)
        yaw = p.getEulerFromQuaternion(orn)[2]

        p.resetDebugVisualizerCamera(
            cameraDistance=4.0,
            cameraYaw=np.degrees(yaw),
            cameraPitch=-45,
            cameraTargetPosition=pos
        )

    def lap_checker(self):
        inside = self._accross_goal()

        lap_completed = False

        if not inside:
            self.left_start = True

        if inside and not self.goal_prev_inside:
            if self.left_start:
                self.lap_count += 1
                self.total_lap_count += 1
                lap_completed = True

        self.goal_prev_inside = inside
        return lap_completed

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
        self.sim_time = 0.0

        self.goal_prev_inside = False
        self.left_start = False
        self.lap_count = 0
        self.start_time = time.time()

        obs, _, _ = self._get_obs()

        for i in range(50):
            p.stepSimulation()

        return obs, {}

    def step(self, action):
        terminated = False
        truncated = False
        info = {}

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
        self.sim_time += self.time_step
        # print(f"hit_data : {self.car.checkHit(self.obj_dict)}")

        goal_inside = self._accross_goal()

        if goal_inside and not self.goal_prev_inside:
            now = time.time()
            self.goal_prev_inside = True

            if not self.lap_started:
                self.lap_started = True
                self.start_time = now
                print("Lap start")
            else:
                lap_time = now - self.start_time
                if self.render:
                    print(f"Lap time : {lap_time:.2f}")
                self.start_time = now

        elif not goal_inside and self.goal_prev_inside:
            self.goal_prev_inside = False

        # print(f"goal_inside : {goal_inside}, "
        #       f"goal_prev_inside : {self.goal_prev_inside}")

        obs, sensor, pos = self._get_obs()
        reward = self._calc_reward(obs, pos, steer, sensor)

        off_all_wheels = self.car.is_all_wheels_off(self.track_id)
        if off_all_wheels:
            self.off_ground_count += 1
        else:
            self.off_ground_count = 0

        course_out = self.off_ground_count > 20
        # course_out = False
        lap_completed = self.lap_checker()

        if lap_completed:
            reward += 100.0
            print(f"# {self.lap_count} lap completed !")

        terminated = course_out or lap_completed

        if terminated:
            reward -= 50.0
            if course_out:
                print("# terminated with course out.")
            if lap_completed:
                print("# terminated with Lap completed.")

        if self.sim_time >= self.max_time:
            truncated = True
            print("# truncated")

        if self.render:
            if not self.car.is_all_wheels_off(self.track_id):
                # print(f"sim time: {self.sim_time:2.2f}")
                self._update_cam_pos()
                # self.car.draw_car_info(throttle, brake, steer)
                # pass

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.engine_id is not None:
            p.disconnect()
            self.engine_id = None
