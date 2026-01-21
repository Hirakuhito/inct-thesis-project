import time
from pathlib import Path

import pybullet as p

from envs.race_env import RacingEnv

from . import config

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent


def main():
    car_pos = [config.CAR["base_x"], config.CAR["base_y"], config.CAR["base_z"]]
    car_orn = p.getQuaternionFromEuler([0, 0, 0])

    env = RacingEnv(car_pos, car_orn, render=True)
    obs, _ = env.reset()

    done = False
    while True:
        action = env.get_baseline_action()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done:
            obs, _ = env.reset()
        time.sleep(1. / 240.)

    env.close()


if __name__ == "__main__":
    main()
