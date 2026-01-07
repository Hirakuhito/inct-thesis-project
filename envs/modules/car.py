import numpy as np
import pybullet as p


class Car:
    def __init__(self, car_id):
        self.car_id = car_id

        self.steer_joints = []
        self.drive_joints = []

        self._get_joints_info("steer")
        self._get_joints_info("wheel")

        self.sensor_front = {
            "origin": np.array([0.0, 0.5, 0.0]),
            "base_direction": np.array([0.0, 1.0, -0.05]),
            "length": 2.5,
            "fov": np.deg2rad(120),
            "num_rays": 7
        }

        self.sensor_right = {
            "origin": np.array([0.2, 0.0, 0.0]),
            "base_direction": np.array([0.5, 0.0, -0.1]),
            "length": 1.5,
            "fov": np.deg2rad(60),
            "num_rays": 3
        }

        self.sensor_left = {
            "origin": np.array([-0.2, 0.0, 0.0]),
            "base_direction": np.array([-0.5, 0.0, -0.1]),
            "length": 1.5,
            "fov": np.deg2rad(60),
            "num_rays": 3
        }

        self.sensor_back = {
            "origin": np.array([0.0, -0.4, 0.0]),
            "base_direction": np.array([0.0, -1.0, -0.1]),
            "length": 1.5,
            "fov": np.deg2rad(120),
            "num_rays": 5
        }

        self.sensors = [
            self.sensor_front,
            self.sensor_left,
            self.sensor_right,
            self.sensor_back,
        ]
    
    def _get_joints_info(self, tag):
        for i in range(p.getNumJoints(self.car_id)):
            info = p.getJointInfo(self.car_id, i)
            name = info[1].decode("utf-8")

            if tag in name:
                self.steer_joints.append(i)
