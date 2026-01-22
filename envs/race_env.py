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

        # Vel(vx, vy, vz), Sensor(22),
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
        self.lap_times = []

        self.max_torque = config.CAR["max_torque"]
        self.max_brake_force = config.CAR["max_brake_force"]
        self.max_steer_angle = config.CAR["max_steer_angle"]

        self.lap_started = False
        self.start_time = None
        self.goal_prev_inside = False
        self.left_start = False

        # ログ表示用
        self.total_episodes = 0
        self.out_of_course_count = 0
        self.timeout_count = 0
        self.total_lap_count = 0
        self.last_info = {}
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

    def get_baseline_action(self):
        # 車両位置
        pos, _ = p.getBasePositionAndOrientation(self.car.car_id)

        # 最も近いコース方向
        nn_idx, _ = self.get_nn_index(pos)
        tangent = -np.array(self.track_normal_vec[nn_idx])
        tangent = tangent[:2]
        tangent = tangent / np.linalg.norm(tangent)

        # 車両前方向
        car_forward = self.get_car_dir()[:2]

        # 向きの角度誤差
        cross = np.cross(
            np.append(car_forward, 0.0),
            np.append(tangent, 0.0)
        )[2]
        dot = np.dot(car_forward, tangent)
        angle_error = np.arctan2(cross, dot)

        # steer 正規化（最大許容角）
        max_angle = 0.5  # rad ≒ 28.6°
        steer = np.clip(angle_error / max_angle, -1.0, 1.0)

        # フルスロットル
        throttle = 1.0
        brake = 0.0

        return np.array([throttle, brake, steer], dtype=np.float32)

    def _get_obs(self):
        vel, _ = p.getBaseVelocity(self.car.car_id)

        # [右前→左前]→[左前→左後ろ]．．．
        sensor = self.car.checkHit(self.obj_dict)

        front = sensor[0]
        f_half = len(front) // 2

        rear = sensor[3]
        r_half = len(rear) // 2

        l_f = sensor[0][f_half:]
        l_m = sensor[1]
        l_r = sensor[3][:r_half]
        r_f = sensor[0][:f_half][::-1]
        r_m = sensor[3][::-1]
        r_r = sensor[2][r_half:][::-1]

        sensor_left = np.array(l_f + l_m + l_r, dtype=np.float32)
        sensor_right = np.array(r_f + r_m + r_r, dtype=np.float32)

        obs = np.concatenate([
             np.array(vel),
             sensor_left,
             sensor_right,
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

    def get_nn_index(self, pos):
        points = self.center_point
        car_pos = np.array(pos[:2]) + np.array([0, 0.5])

        diff = points - car_pos
        dists = np.linalg.norm(diff, axis=1)

        idx = int(np.argmin(dists))
        idx_next = (idx - 10) % len(points)

        return idx, idx_next

    def get_runoff_ratio(self, sensor):
        runoff_count = 0.0

        for i in sensor:
            if i == -1:
                runoff_count += 1

        runoff_ratio = runoff_count / len(sensor)

        return runoff_ratio

    def get_car_dir(self):
        pos, orn = p.getBasePositionAndOrientation(self.car.car_id)

        rot_mat = p.getMatrixFromQuaternion(orn)
        rot_mat = np.array(rot_mat).reshape(3, 3)

        forward_local = np.array([0, 1, 0])
        forward_world = rot_mat @ forward_local

        forward_world = forward_world / np.linalg.norm(forward_world)

        return forward_world

    def _calc_reward(self, obs, action, pos):
        reward = 0.0

        # ======================
        # 観測の分解
        # ======================
        vel_dim = 3
        sensor_dim = (len(obs) - vel_dim) // 2

        sensor_left = obs[vel_dim:(vel_dim + sensor_dim)]
        sensor_right = obs[(vel_dim + sensor_dim):(vel_dim + 2 * sensor_dim)]

        throttle, brake, steer = action
        steer_dir = np.sign(steer)
        steer_abs = abs(steer)

        car_vel = np.array(obs[:2])

        # ======================
        # コース情報
        # ======================
        nn_idx, nn_idx_next = self.get_nn_index(pos)
        tangent_near = -np.array(self.track_normal_vec[nn_idx])
        tangent_far = -np.array(self.track_normal_vec[nn_idx_next])

        car_forward = self.get_car_dir()[:2]

        dot_near = np.dot(tangent_near, car_forward)
        dot_far = np.dot(tangent_far, car_forward)

        forward_speed = np.dot(tangent_near, car_vel)
        speed_scale = np.clip(forward_speed / 5.5, 0.0, 1.0)

        # ======================
        # コース曲率
        # ======================
        tan_dot = np.dot(tangent_near, tangent_far)
        theta = np.arccos(np.clip(tan_dot, -1.0, 1.0))
        curve_strength = np.clip(theta / self.course_theta_max, 0.0, 1.0)

        # ======================
        # 前進報酬
        # ======================
        if forward_speed > 0:
            forward_reward = (forward_speed ** 2) * 5.0
        else:
            forward_reward = 0.0

        # ======================
        #  バックペナルティ
        # ======================
        back_penalty = 0.0
        if forward_speed < 0:
            back_penalty = forward_speed * 5.0  # 強めに抑制

        # ======================
        #  向きズレペナルティ
        # ======================
        dir_error = (1 - dot_near) * 0.75 + (1 - dot_far) * 0.25
        direction_penalty = -(
            dir_error
            * speed_scale
            * (1 - curve_strength)
            * 3.0
        )

        # ======================
        # ステア制御
        # ======================

        # カーブ強度
        curve_eps = 0.05  # 直線判定
        need_steer = curve_strength > curve_eps

        steer_reward = 0.0
        steer_penalty = 0.0

        if not need_steer:
            # 直線：ステア 即ペナルティ
            steer_penalty = - (steer_abs ** 2) * speed_scale * 4.0
        else:
            # カーブ：適正量のみ許可
            target = curve_strength
            steer_error = abs(steer_abs - target)

            steer_reward = np.exp(-steer_error * 5.0) * speed_scale * 2.0
            steer_penalty = - (steer_error ** 2) * speed_scale * 3.0

        # ======================
        #  タイヤ接地ペナルティ
        # ======================
        wheel_contacts = self.car.get_wheel_contact(self.track_id)
        contact_penalty = 0.0
        for c in wheel_contacts:
            if not c:
                contact_penalty -= speed_scale * 2.0

        # ======================
        # センサーペナルティ
        # ======================
        danger_left = self.get_runoff_ratio(sensor_left)
        danger_right = self.get_runoff_ratio(sensor_right)

        if steer_dir > 0:
            danger = danger_right
        elif steer_dir < 0:
            danger = danger_left
        else:
            danger = 0.5 * (danger_left + danger_right)

        danger_level = np.tanh((danger - 0.3) * 6.0)
        danger_level = np.clip(danger_level, 0.0, 1.0)

        sensor_penalty = -danger_level * speed_scale * 5.0

        # ======================
        # 総和
        # ======================
        reward = (
            forward_reward
            + back_penalty
            + direction_penalty
            + steer_reward
            + steer_penalty
            + contact_penalty
            + sensor_penalty
        )

        info = {
            "forward": forward_reward,
            "back": back_penalty,
            "direction": direction_penalty,
            "steer": steer_reward,
            "steer_penalty": steer_penalty,
            "contact": contact_penalty,
            "sensor": sensor_penalty,
        }

        return reward, info

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
        if hasattr(self, "last_info"):
            self.total_episodes += 1

            if self.last_info.get("termination") == "out_of_course":
                self.out_of_course_count += 1
            elif self.last_info.get("termination") == "lap_completed":
                self.total_lap_count += 1
            elif self.last_info.get("truncate") == "timeout":
                self.timeout_count += 1

        self.last_info = {}

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

        obs = self._get_obs()

        for i in range(50):
            p.stepSimulation()

        return obs, {}

    def step(self, action):
        # 中断・切り捨て用フラグ
        terminated = False
        truncated = False

        # 車両の出力
        throttle, brake, steer = action

        info = {}

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

        obs = self._get_obs()
        pos, _ = p.getBasePositionAndOrientation(self.car.car_id)

        if not np.all(np.isfinite(obs)):
            print("obs =", obs)
            raise RuntimeError("NaN in observation")

        reward, reward_info = self._calc_reward(obs, action, pos)

        info = {
            "reward/forward": reward_info["forward"],
            "reward/back": reward_info["back"],
            "reward/direction": reward_info["direction"],
            "reward/steer": reward_info["steer"],
            "reward/steer_penalty": reward_info["steer_penalty"],
            "reward/contact": reward_info["contact"],
            "reward/sensor": reward_info["sensor"],
        }

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
                reward -= 50.0
                info["termination"] = "out_of_course"
                print("# terminated with course out.")
            if lap_completed:
                reward += 50.0
                info["termination"] = "lap_completed"
                print("# terminated with Lap completed.")

        if self.sim_time >= self.max_time:
            truncated = True
            info["truncate"] = "timeout"
            print("# truncated")

        if self.render:
            if not self.car.is_all_wheels_off(self.track_id):
                # print(f"sim time: {self.sim_time:2.2f}")
                self._update_cam_pos()
                # pass
                # self.car.draw_car_info(throttle, brake, steer)

        self.last_info = info

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.engine_id is not None:
            p.disconnect()
            self.engine_id = None
