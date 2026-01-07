import numpy as np
import pybullet as p
from pathlib import Path


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
        current_path = Path(__file__).resolve()
        car_directory = "formular/formular_car/car.urdf"
        car_path = str(current_path.parent.parent / car_directory)

        self.car_id = p.loadURDF(
            car_path,
            basePosition=self.base_pos,
            baseOrientation=self.base_orn,
            globalScaling=0.2
        )

    def _get_joints_info(self):
        for i in range(p.getNumJoints(self.car_id)):
            info = p.getJointInfo(self.car_id, i)
            name = info[1].decode("utf-8")

            if "steer" in name:
                self.steer_joints.append(i)
            if "wheel" in name:
                self.wheel_joints.append(i)

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

            dirs_local = self.gen_world_direction(
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

    def checkHit(self):
        all_rays = self.local2world()

        starts = []
        ends = []
        ray_map = []

        for sensor_index, rays in enumerate(all_rays):
            for ray_index, (s, e) in enumerate(rays):
                # p.addUserDebugLine(s, e, [1, 0, 0], 4, 0.1)
                starts.append(s)
                ends.append(e)
                ray_map.append((sensor_index, ray_index))

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

    def reset(self):
        pass

    def action(self):
        pass
