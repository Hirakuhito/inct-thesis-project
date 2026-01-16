import numpy as np

RENDER = False

MAX_TIME = 20
TOTAL_TIME_STEP = 100_000
N_SPLIT = 100
SAVE_FREQ = TOTAL_TIME_STEP // N_SPLIT

TARGET_LAP = 10

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
    "max_torque": 200.0,
    "max_brake_force": 1000.0,
    "max_steer_angle": np.pi / 3
}

FRICTION = {
    "body": 1.0,
    "track": 2.5,
    "runoff": 1.0,
    "lateral": 2.0,
    "rolling": 0.01,
    "spining": 0.01
}
