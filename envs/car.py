import pprint
from pathlib import Path

import numpy as np
import pybullet as p

from main import config

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent


class Car:
    def __init__(self, pos, orn):
        self.base_pos = pos
        self.base_orn = orn

        self.car_id = None

        self.steer_joints = [0, 2]
        self.wheel_joints = []

        self._setup_car()
        self._get_joints_info()
        self._set_wheel_dynamics()

        self.sensor_f = {
            "origin": np.array([0.0, 0.5, 0.0]),
            "base_direction": np.array([0.0, 1.0, -0.1]),
            "length": 1.0,
            "fov": np.deg2rad(120),
            "num_rays": 7
        }

        self.sensor_r = {
            "origin": np.array([0.2, 0.0, 0.0]),
            "base_direction": np.array([0.5, 0.0, -0.08]),
            "length": 1.0,
            "fov": np.deg2rad(60),
            "num_rays": 3
        }

        self.sensor_l = {
            "origin": np.array([-0.2, 0.0, 0.0]),
            "base_direction": np.array([-0.5, 0.0, -0.1]),
            "length": 1.0,
            "fov": np.deg2rad(60),
            "num_rays": 3
        }

        self.sensor_b = {
            "origin": np.array([0.0, -0.4, 0.0]),
            "base_direction": np.array([0.0, -1.0, -0.1]),
            "length": 1.0,
            "fov": np.deg2rad(120),
            "num_rays": 5
        }

        self.sensors = [
            self.sensor_f,
            self.sensor_l,
            self.sensor_r,
            self.sensor_b
        ]

        self.wheel_sign = {
            1: 0,  # front right
            3: 0,  # front left
            5: -1,  # rear right
            7: -1,  # rear left
        }

    def _setup_car(self):
        car_path = str(PROJECT_ROOT / config.CAR["path"] / config.CAR["urdf"])

        self.car_id = p.loadURDF(
            car_path,
            basePosition=self.base_pos,
            baseOrientation=self.base_orn,
            globalScaling=config.CAR["scale"]
        )

        p.changeDynamics(
            self.car_id,
            -1,
            lateralFriction=config.FRICTION["body"]
        )

    def _set_wheel_dynamics(self):
        for j in self.wheel_joints:
            p.changeDynamics(
                self.car_id,
                j,
                lateralFriction=config.FRICTION["lateral"],
                rollingFriction=config.FRICTION["rolling"],
                spinningFriction=config.FRICTION["spining"],
            )

        for j in self.wheel_joints:
            p.setJointMotorControl2(
                self.car_id,
                j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )

    def _get_joints_info(self):
        for i in range(p.getNumJoints(self.car_id)):
            info = p.getJointInfo(self.car_id, i)
            name = info[1].decode("utf-8")

            # print(f"info : {info}")

            if "wheel" in name:
                self.wheel_joints.append(i)
            print(f"joint[{i}]: {info[13]}")
        pprint.pprint(f"steer joints index : {self.steer_joints}")
        pprint.pprint(f"wheel joints index : {self.wheel_joints}")

    def _gen_world_direction(self, base_dir, fov, num):
        angles = np.linspace(-fov/2, fov/2, num)
        base_dir = base_dir / np.linalg.norm(base_dir)

        dirs = []
        for a in angles:
            rot = np.array([
                [np.cos(a), -np.sin(a), 0],
                [np.sin(a),  np.cos(a), 0],
                [0,        0,       1],
            ])
            dirs.append(rot @ base_dir)

        return dirs

    def _local2world(self):
        pos, orn = p.getBasePositionAndOrientation(self.car_id)
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        all_rays = []

        for sensor in self.sensors:
            rays = []

            dirs_local = self._gen_world_direction(
                sensor["base_direction"],
                sensor["fov"],
                sensor["num_rays"]
            )

            origin_world = np.array(pos) + rot @ sensor["origin"]

            for d in dirs_local:
                dir_world = rot @ d
                end_world = origin_world + dir_world * sensor["length"]

                rays.append((origin_world.tolist(), end_world.tolist()))

            all_rays.append(rays)

        return all_rays

    def draw_car_info(self, throttle, brake, steer):
        pos, orn = p.getBasePositionAndOrientation(self.car_id)
        base = np.array(pos) + np.array([0, 0, 1.0])

        dy = 0.2

        p.addUserDebugText(
            f"Throttle: {throttle:.2f}",
            (base + np.array([0, 0, 0])).tolist(),
            textColorRGB=[1, 1, 1],
            lifeTime=0.3,
            textSize=1.1
        )
        p.addUserDebugText(
            f"Brake   : {brake:.2f}",
            (base + np.array([0, 0, -dy])).tolist(),
            textColorRGB=[1, 1, 1],
            lifeTime=0.2,
            textSize=1.1
        )
        p.addUserDebugText(
            f"Steer   : {steer:.2f}",
            (base + np.array([0, 0, -2*dy])).tolist(),
            textColorRGB=[1, 1, 1],
            lifeTime=0.1,
            textSize=1.1
        )

    def is_out_of_track(self, obs):
        sensor = obs[6:]
        min_dist = sensor.min()
        return min_dist < 0.05

    def get_wheel_contact(self, ground_id):
        contacts = []

        for wheel in self.wheel_joints:
            is_contact = False

            pts = p.getContactPoints(
                bodyA=self.car_id,
                bodyB=ground_id,
                linkIndexA=wheel
            )

            if len(pts) > 0:
                is_contact = True

            contacts.append(is_contact)

        return contacts

    def is_all_wheels_off(self, ground_id):
        contacts = self.get_wheel_contact(ground_id)
        return not any(contacts)

    def checkHit(self, obj_dict):
        all_rays = self._local2world()

        starts = []
        ends = []
        ray_map = []

        for sensor_index, rays in enumerate(all_rays):
            for ray_index, (s, e) in enumerate(rays):
                starts.append(s)
                ends.append(e)
                ray_map.append((sensor_index, ray_index))
                p.addUserDebugLine(s, e, [1, 0, 0], 1, 0.1)

            # print(f"starts: {starts}")
            # print(f"ends: {ends}")

        results = p.rayTestBatch(starts, ends)
        # print(f"result : {results}")

        hit_data = [
            [] for _ in range(len(all_rays))
        ]

        for (sensor_index, _), hit in zip(ray_map, results):
            hit_object_uid = hit[0]
            # print(f"hit_object_uid : {hit_object_uid}")

            if hit_object_uid == obj_dict["track"]:
                hit_data[sensor_index].append(1.0)
            elif hit_object_uid == obj_dict["runoff"]:
                hit_data[sensor_index].append(-1.0)
            else:
                hit_data[sensor_index].append(0.0)

        return hit_data

    def reset(self, pos, orn):
        p.resetBasePositionAndOrientation(
            self.car_id,
            pos,
            orn
        )
        p.resetBaseVelocity(self.car_id, [0, 0, 0], [0, 0, 0])

    def apply_action(self, throttle, brake, steer, max_torque,
                     max_brake, max_steer):
        steer_angle = steer * max_steer

        for j in self.steer_joints:
            p.setJointMotorControl2(
                self.car_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=steer_angle
            )

        for j in self.wheel_joints:
            wheel_state = p.getJointState(self.car_id, j)
            wheel_vel = wheel_state[1] * self.wheel_sign[j]

            drive_torque = throttle * max_torque

            brake_torque = 0.0
            if abs(wheel_vel) > 1e-2 and brake > 1e-3:
                brake_torque = brake * max_brake * (np.sign(wheel_vel))

            total_torque = drive_torque + brake_torque
            force = self.wheel_sign[j] * total_torque
            # print(
            #     f"total torque[{j}] : {drive_torque:5.2f} - "
            #     f"{brake_torque:5.2f} = {total_torque:5.2f}"
            # )

            p.setJointMotorControl2(
                self.car_id,
                j,
                p.TORQUE_CONTROL,
                force=force
            )
