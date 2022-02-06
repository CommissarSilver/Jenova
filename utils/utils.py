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
            double_q=True,
            learning_starts=1000,
            target_network_update_freq=500,
            prioritized_replay=False,
            prioritized_replay_alpha=0.6,
            prioritized_replay_beta0=0.4,
            prioritized_replay_beta_iters=None,
            prioritized_replay_eps=1e-06,
            param_noise=False,
            n_cpu_tf_sess=None,
            verbose=0,
            tensorboard_log=None,
            _init_setup_model=True,
            policy_kwargs=None,
            full_tensorboard_log=False,
            seed=None,
        )

    return model

