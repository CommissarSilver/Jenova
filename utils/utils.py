from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv


def create_model(algorithm: str, environment):
    supported_algorithms = ["DQN"]
    if algorithm.upper() == "DQN":
        from stable_baselines3.dqn.dqn import DQN
        from stable_baselines3.dqn.policies import MlpPolicy

        model = DQN(
            MlpPolicy,
            environment,
            gamma=0.90,
            learning_rate=0.0005,
            buffer_size=10000,
            exploration_fraction=1,
            exploration_final_eps=0.02,
            exploration_initial_eps=1.0,
            train_freq=1,
            batch_size=32,
            learning_starts=1000,
            target_update_interval=500,
            verbose=0,
            _init_setup_model=True,
            policy_kwargs=None,
            seed=None,
        )

    return model


def load_model(algorithm: str, environment, model_path: str):
    supported_algorithms = ["DQN"]
    if algorithm.upper() == "DQN":
        from stable_baselines3.dqn.dqn import DQN

        model = DQN.load(model_path)
        model.set_env(environment)

    return model
