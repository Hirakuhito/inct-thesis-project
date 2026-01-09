import time

import numpy as np
import pybullet as p

from envs.race_env import RacingEnv

from . import config


def main():
    car_pos = [config.CIRCUIT["radius"], 0, 0.1]
    car_orn = p.getQuaternionFromEuler([0, 0, 0])

    env = RacingEnv(car_pos, car_orn, render=True)
    obs, _ = env.reset()

    while True:
        action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        time.sleep(1. / 240.)

        keys = p.getKeyboardEvents()

        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERD:
            break

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
