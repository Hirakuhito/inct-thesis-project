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
        self.lap_times = []

        self.max_torque = config.CAR["max_torque"]
        self.max_brake_force = config.CAR["max_brake_force"]
        self.max_steer_angle = config.CAR["max_steer_angle"]

        self.lap_started = False
        self.start_time = None
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

        thetas = []
        normal_len = len(self.track_normal_vec)
        for i in range(len(self.track_normal_vec) - 1):
            dot_raw = np.dot(
                self.track_normal_vec[i],
                self.track_normal_vec[(i-10) % normal_len]
            )
            dot = np.clip(
                dot_raw,
                -1.0,
                1.0
            )
            thetas.append(np.arccos(dot))

        thetas = np.array(thetas)
        self.course_theta_max = np.max(thetas)

        self.close()
        self.engine_id = p.connect(
            p.GUI if self.render else p.DIRECT,
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

    def get_car_dir(self):
        pos, orn = p.getBasePositionAndOrientation(self.car.car_id)

        rot_mat = p.getMatrixFromQuaternion(orn)
        rot_mat = np.array(rot_mat).reshape(3, 3)

        forward_local = np.array([0, 1, 0])
        forward_world = rot_mat @ forward_local

        forward_world = forward_world / np.linalg.norm(forward_world)

        return forward_world

    def _calc_reward(self, obs, pos, action, sensor):
        reward = 0.0
        throttle, brake, steer = action
        # コースの総点数の取得，進捗率用
        # course_length = len(self.center_point)

        # 車両の速さと速度の計算
        car_vel = np.array(obs[:2])
        # car_speed = np.linalg.norm(car_vel)

        # 最も近い方向ベクトルと，少し先の方向ベクトルのインデックスを取得
        nn_idx, nn_indx_next = self.get_nn_index(pos)
        idx_point = self.track_normal_vec[nn_idx]
        idx_point_next = self.track_normal_vec[nn_indx_next]

        # 車の前方向ベクトルの取得
        car_forward = self.get_car_dir()[:2]

        # 車とコースの向きの一致度を計算
        tangent_near = np.array(idx_point) * -1
        tangent_far = np.array(idx_point_next) * -1
        dot_near = np.dot(tangent_near, car_forward)
        dot_far = np.dot(tangent_far, car_forward)

        # コース前方に対する車両の速度の計算
        forward_speed = np.dot(tangent_near, car_vel)
        speed_scale = np.clip(forward_speed / 5.5, 0.0, 1.0)

        # コースに対するホイールの接地判定取得
        wheel_contacts = self.car.get_wheel_contact(self.track_id)

        # センサー 前方，側方，後方 のランオフ率の取得
        r_f, r_s, r_b = self.get_runoff_ratio(sensor)

        # 後ろを向いている・前向きでバックに対するペナルティ
        back_penalty = 0.0
        if dot_near < 0 or forward_speed < 0:
            back_penalty = dot_near * 5.0

        # 進行方向の不一致度に対するペナルティ
        dir_fusion = (1 - dot_near) * 0.7 + (1 - dot_far) * 0.3
        dir_penalty = -dir_fusion * speed_scale * 3.0

        # ホイールの接地率に対するペナルティー
        wheel_contact_penalty = 0.0
        for c in wheel_contacts:
            if not c:
                wheel_contact_penalty -= forward_speed

        # ランオフ検出率に対するペナルティ
        fusion_sensor = (r_f * 0.6 + r_s * 0.3 + r_b * 0.1)
        if fusion_sensor < 0.1:
            sensor_penalty = 0.0
        sensor_penalty = (
            -np.tanh(fusion_sensor) * max(speed_scale - 0.3, 0.0)
            * 3.0
        )

        # 少し先の方向ベクトルと最近のベクトルとの角度差
        tan_dot = np.dot(tangent_far, tangent_near)
        theta = np.arccos(np.clip(tan_dot, -1.0, 1.0))
        curve_strength = np.clip(theta / self.course_theta_max, 0.0, 1.0)
        steer_norm = abs(steer)

        # コースの曲率とステア量の不一致度の計算
        target_steer = curve_strength ** 0.5
        mismatch = - abs(target_steer - steer_norm) ** 2
        # excess = -max(steer_norm - curve_strength, 0.0) ** 2

        # コースの曲率に対するステアリング量のペナルティ
        steer_penalty = mismatch * speed_scale * 2.0

        # スピードに対する報酬
        forward_speed_reward = 0.0
        if forward_speed < 0.1:
            forward_speed_reward = - min(self.sim_time, 5.0)
            print("now I'm stopping...")
        else:
            forward_speed_reward = forward_speed**2

        reward = (
            forward_speed_reward
            + dir_penalty
            + back_penalty
            + wheel_contact_penalty
            + steer_penalty
            + sensor_penalty
        )

        # print(
        #     f"forward_speed: {forward_speed:.2f}  "
        #     f"steer_penalty:{steer_penalty:.2f}  "
        #     f"sensor_penalty:{sensor_penalty:.2f}  "
        #     f"sim_time:{self.sim_time:.2f}  "
        #     f"reward:{reward:.2f}"
        # )

        if not np.isfinite(reward):
            print("reward invalid:", reward)
            reward = -100.0

        # if self.render:
        #     print(f"reward:{reward:.2f}")
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
        lap_time = None

        if not inside:
            self.left_start = True

        if inside and not self.goal_prev_inside and self.left_start:
            now = time.time()

            if not self.lap_started:
                self.lap_started = True
                self.start_time = now
                print("# Lap start")
            else:
                lap_time = now - self.start_time
                self.lap_times.append(lap_time)
                self.start_time = now

                self.lap_count += 1
                self.total_lap_count += 1
                lap_completed = True

        self.goal_prev_inside = inside
        return lap_completed, lap_time

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

        self.lap_started = False
        self.start_time = None

        obs, _, _ = self._get_obs()

        for i in range(50):
            p.stepSimulation()

        return obs, {}

    def step(self, action):
        # 中断・切り捨て用フラグ
        terminated = False
        truncated = False
        info = {}

        # 車両の出力
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

        obs, sensor, pos = self._get_obs()

        if not np.all(np.isfinite(obs)):
            print("obs =", obs)
            raise RuntimeError("NaN in observation")

        reward = self._calc_reward(obs, pos, action, sensor)

        lap_completed, lap_time = self.lap_checker()

        if lap_completed:
            print(f"# {self.total_lap_count} lap completed !")

            if self.render and lap_time is not None:
                avg = sum(self.lap_times) / len(self.lap_times)
                print(f"Lap time: {lap_time:.2f}s | Avg: {avg:.2f}s")

        off_all_wheels = self.car.is_all_wheels_off(self.track_id)
        if off_all_wheels:
            self.off_ground_count += 1
        else:
            self.off_ground_count = 0

        course_out = self.off_ground_count > 20
        # course_out = Fals

        terminated = course_out or lap_completed

        if terminated:
            if course_out:
                reward -= 1.0
                print("# terminated with course out.")
            if lap_completed:
                reward += 10.0
                print("# terminated with Lap completed.")

        if self.sim_time >= self.max_time:
            truncated = True
            print("# truncated")

        if self.render:
            if not self.car.is_all_wheels_off(self.track_id):
                # print(f"sim time: {self.sim_time:2.2f}")
                self._update_cam_pos()
                # self.car.draw_car_info(throttle, brake, steer)

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.engine_id is not None:
            p.disconnect()
            self.engine_id = None
