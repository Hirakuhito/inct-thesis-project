from datetime import datetime
from pathlib import Path

import pybullet as p
from stable_baselines3 import PPO

from envs.race_env import RacingEnv

from . import config

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent


def main():
    car_pos = [config.CIRCUIT["radius"], 0, 0.1]
    car_orn = p.getQuaternionFromEuler([0, 0, 0])
    env = RacingEnv(car_pos, car_orn, render=config.RENDER)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tb_logs/"
    )

    model.learn(
        total_timesteps=config.TOTAL_TIME_STEP,
        tb_log_name="ML_v1"
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"model_{timestamp}"
    file_path = PROJECT_ROOT / "models" / filename
    model.save(str(file_path))


if __name__ == "__main__":
    main()
