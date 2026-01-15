from datetime import datetime
from pathlib import Path

import pybullet as p
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.race_env import RacingEnv

from . import config

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent


def main():
    checkpoint_dir, best_model_dir, \
        eval_log_dir, tb_log_dir = gen_exp_data_dir()

    car_pos = [
        config.CAR["base_x"],
        config.CAR["base_y"],
        config.CAR["base_z"]
    ]
    car_orn = p.getQuaternionFromEuler([0, 0, 0])
    env = RacingEnv(car_pos, car_orn, render=config.RENDER)
    env = TimeLimit(env, max_episode_steps=1000)
    env = Monitor(env)

    eval_env = RacingEnv(car_pos, car_orn, render=False)
    eval_env = TimeLimit(eval_env, max_episode_steps=5000)
    eval_env = Monitor(eval_env)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        verbose=1,
        tensorboard_log=str(tb_log_dir),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=config.SAVE_FREQ,
        save_path=str(checkpoint_dir),
        name_prefix="ppo"
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=config.SAVE_FREQ * 5,
        n_eval_episodes=1,
        deterministic=True
    )

    model.learn(
        total_timesteps=config.TOTAL_TIME_STEP,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="ML_v1"
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"model_{timestamp}"
    file_path = PROJECT_ROOT / "models" / filename
    model.save(str(file_path))

    print(f"Training finished. Results saved in: {file_path}")


def gen_exp_data_dir():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = PROJECT_ROOT / "experiments" / f"run_{run_id}"

    checkpoint_dir = base_dir / "checkpoints"
    best_model_dir = base_dir / "best_model"
    eval_log_dir = base_dir / "eval_logs"
    tb_log_dir = base_dir / "tb_logs"

    for d in [checkpoint_dir, best_model_dir, eval_log_dir, tb_log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir, best_model_dir, eval_log_dir, tb_log_dir


if __name__ == "__main__":
    main()
