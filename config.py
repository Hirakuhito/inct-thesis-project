import numpy as np

MAX_STEPS = 10_000

CIRCUIT = {
    "track_path": 'circuitData/track.obj',
    "runoff_path": 'circuitData/runoff.obj',
    "straight": 7,
    "radius": 3,
    "width": 2
}

CAR = {
    "car_path": 'formular/formular_car/car.urdf',
    "scale": 0.2,
    "max_engine_force": 1200.0,
    "max_torque": 100.0,
    "max_brake_force": 100.0,
    "max_steer_angle": np.pi / 3
}

