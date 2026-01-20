import numpy as np

RENDER = False

MAX_TIME = 20
STEP_PER_ENV = 10_000
TOTAL_TIME_STEP = 8 * STEP_PER_ENV
N_SPLIT = 10
SAVE_FREQ = TOTAL_TIME_STEP // N_SPLIT

TARGET_LAP = 10

GRAVITY = [0, 0, -9.8]

CIRCUIT = {
    "path": 'assets/circuitData',
    "track": 'track2',
    "runoff": 'runoff2',
    "straight": 10,
    "radius": 4,
    "width": 2
}

CAR = {
    "path": 'assets/formular/formular_car',
    "urdf": 'car.urdf',
    "scale": 1,
    "base_x": CIRCUIT["radius"],
    "base_y": -0.6,
    "base_z": 0.1,
    "max_torque": 1.0,
    "max_brake_force": 1.0,
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
