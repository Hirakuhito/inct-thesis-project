import numpy as np

MAX_STEPS = 10_000
TOTAL_TIME_STEP = 50_000
GRAVITY = [0, 0, -9.8]

CIRCUIT = {
    "path": 'assets/circuitData',
    "track": 'track',
    "runoff": 'runoff',
    "straight": 7,
    "radius": 3,
    "width": 2
}

CAR = {
    "path": 'assets/formular/formular_car',
    "urdf": 'car.urdf',
    "scale": 0.2,
    "max_engine_force": 1200.0,
    "max_torque": 100.0,
    "max_brake_force": 100.0,
    "max_steer_angle": np.pi / 3
}
