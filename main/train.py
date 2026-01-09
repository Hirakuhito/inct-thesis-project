import pybullet as p

from envs.race_env import RacingEnv

from . import config
from stable_baselines3 import PPO


def main():
    car_pos = [config.CIRCUIT["radius"], 0, 0.1]
    car_orn = p.getQuaternionFromEuler([0, 0, 0])
    env = RacingEnv(car_pos, car_orn, render=True)
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


if __name__ == "__main__":
    main()
