import numpy as np

RENDER = False
MAX_STEPS = 2_000
TOTAL_TIME_STEP = 500_000
N_SPLIT = 10
SAVE_FREQ = TOTAL_TIME_STEP // N_SPLIT
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
    "base_x": CIRCUIT["radius"],
    "base_y": -0.6,
    "base_z": 0.1,
    "max_engine_force": 1200.0,
    "max_torque": 100.0,
    "max_brake_force": 100.0,
    "max_steer_angle": np.pi / 3
}

FRICTION = {
    "body": 1.0,
    "track": 1.5,
    "runoff": 1.0,
    "lateral": 2.0,
    "rolling": 0.01,
    "spining": 0.01
}
