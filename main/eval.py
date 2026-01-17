import time
from pathlib import Path

import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs.race_env import RacingEnv

from . import config

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent


def main():
    car_pos = [
        config.CAR["base_x"],
        config.CAR["base_y"],
        config.CAR["base_z"]
    ]
    car_orn = p.getQuaternionFromEuler([0, 0, 0])

    env = RacingEnv(car_pos, car_orn, render=True)
    env = Monitor(env)

    run_name = "experiments/run_20260117_225826"
    model_path = PROJECT_ROOT / run_name \
        / "best_model" / "best_model"

    model = PPO.load(
        model_path,
        env=env
    )

    obs, _ = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        time.sleep(1. / 240.)

        keys = p.getKeyboardEvents()

        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            break

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
