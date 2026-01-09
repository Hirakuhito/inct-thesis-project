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

        self.steer_joints = []
        self.wheel_joints = []

        self._setup_car()
        self._get_joints_info()

        self.sensor_f = {
            "origin": np.array([0.0, 0.5, 0.0]),
            "base_direction": np.array([0.0, 1.0, -0.05]),
            "length": 2.5,
            "fov": np.deg2rad(120),
            "num_rays": 7
        }

        self.sensor_r = {
            "origin": np.array([0.2, 0.0, 0.0]),
            "base_direction": np.array([0.5, 0.0, -0.1]),
            "length": 1.5,
            "fov": np.deg2rad(60),
            "num_rays": 3
        }

        self.sensor_l = {
            "origin": np.array([-0.2, 0.0, 0.0]),
            "base_direction": np.array([-0.5, 0.0, -0.1]),
            "length": 1.5,
            "fov": np.deg2rad(60),
            "num_rays": 3
        }

        self.sensor_b = {
            "origin": np.array([0.0, -0.4, 0.0]),
            "base_direction": np.array([0.0, -1.0, -0.1]),
            "length": 1.5,
            "fov": np.deg2rad(120),
            "num_rays": 5
        }

        self.sensors = [
            self.sensor_f,
            self.sensor_l,
            self.sensor_r,
            self.sensor_b
        ]

    def _setup_car(self):
        car_path = str(PROJECT_ROOT / config.CAR["path"] / config.CAR["urdf"])

        self.car_id = p.loadURDF(
            car_path,
            basePosition=self.base_pos,
            baseOrientation=self.base_orn,
            globalScaling=config.CAR["scale"]
        )

    def _get_joints_info(self):
        for i in range(p.getNumJoints(self.car_id)):
            info = p.getJointInfo(self.car_id, i)
            name = info[1].decode("utf-8")

            if "steer" in name:
                self.steer_joints.append(i)
            if "wheel" in name:
                self.wheel_joints.append(i)

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
                [0,        0,       0],
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

    def is_out_of_track(self, obs):
        sensor = obs[6:]
        min_dist = sensor.min()
        return min_dist < 0.05

    def get_wheel_contact(self, ground_ids):
        contacts = []

        for wheel in self.wheel_joints:
            is_contact = False

            for ground_id in ground_ids:
                pts = p.getContactPoints(
                    bodyA=self.car_id,
                    bodyB=ground_id,
                    linkIndexA=wheel
                )
                if len(pts) > 0:
                    is_contact = True
                    break

                contacts.append(is_contact)

        return contacts

    def checkHit(self):
        all_rays = self._local2world()

        starts = []
        ends = []
        ray_map = []

        for sensor_index, rays in enumerate(all_rays):
            for ray_index, (s, e) in enumerate(rays):
                # p.addUserDebugLine(s, e, [1, 0, 0], 4, 0.1)
                starts.append(s)
                ends.append(e)
                ray_map.append((sensor_index, ray_index))

                # p.addUserDebugLine(s, e, [1, 0, 0], 1, 0.1)

        results = p.rayTestBatch(starts, ends)

        hit_data = [
            [] for _ in range(len(all_rays))
        ]

        for (sensor_index, _), hit in zip(ray_map, results):
            hit_object_uid = hit[0]
            hit_fraction = hit[2]

            if hit_object_uid < 0:
                hit_data[sensor_index].append(1.0)
            else:
                hit_data[sensor_index].append(hit_fraction)

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

        # drive_force = throttle * max_torque
        # brake_force = brake * max_brake

        for j in self.wheel_joints:
            p.setJointMotorControl2(
                self.car_id,
                j,
                p.VELOCITY_CONTROL,
                targetVelocity=300,
                force=-max_torque
            )
