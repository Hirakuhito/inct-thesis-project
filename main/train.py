from envs.race_env import RacingEnv


def main():
    env = RacingEnv(render=True)

    env.reset()


if __name__ == "__main__":
    main()
