import pybullet as p

from envs.race_env import RacingEnv
from . import config


def main():
    car_pos = [config.CIRCUIT["radius"], 0, 0.1]
    car_orn = p.getQuaternionFromEuler([0, 0, 0])
    env = RacingEnv(car_pos, car_orn, render=True)

    env.reset()

    while True:
        p.stepSimulation()

        keys = p.getKeyboardEvents()
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            break


if __name__ == "__main__":
    main()
