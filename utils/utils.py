from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

import numpy as np


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
    elif algorithm.upper() == "A2C":
        # ======================= HYPER-PARAMS =======================
        # gamma: discount factor
        # learning_rate: learning rate
        # vf_coef: (float) Value function coefficient for the loss calculation
        # ent_coef: (float) Entropy coefficient for the loss calculation
        # max_grad_norm: (float) The maximum value for the gradient clipping
        # alpha: (float) RMSProp decay parameter (default: 0.99)
        # momentum: (float) RMSProp momentum parameter (default: 0.0)
        # epsilon: (float) RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)
        # lr_schedule: (str) The type of scheduler for the learning rate update (‘linear’, ‘constant’, ‘double_linear_con’, ‘middle_drop’ or ‘double_middle_drop’)
        # ======================= HYPER-PARAMS =======================
        from stable_baselines3.a2c.a2c import A2C
        from stable_baselines3.a2c.policies import MlpPolicy

        env = DummyVecEnv([lambda: environment])
        model = A2C(
            MlpPolicy,
            env,
            gamma=0.90,
            learning_rate=0.0005,
            verbose=0,
            tensorboard_log=None,
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
    if algorithm.upper() == "A2C":
        from stable_baselines3.a2c.a2c import A2C

        model = A2C.load(model_path)
        model.set_env(environment)

    return model

# TODO - UNDERSTAND THIS MODULE


def test_agent(env, model_path: str, algo, mode):
    agent_actions = []
    print("Evaluation of an agent from " + model_path)
    model = load_model(algo, env, model_path)
    try:
        if model:
            if mode.upper() == "PAIRWISE" and algo.upper() != "DQN":
                env = model.get_env()
                obs = env.reset()
                done = False
                while True:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, done, info = env.step(action)
                    if done:
                        break
                return env.get_attr("sorted_test_cases_vector")[0]

            elif mode.upper() == "PAIRWISE" and algo.upper() == "DQN":
                env = model.get_env()
                obs = env.reset()
                done = False
                while True:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, done, info = env.step(action)
                    if done:
                        break
                return env.sorted_test_cases_vector

            elif mode.upper() == "POINTWISE":
                test_cases = env.cycle_logs.test_cases

                if algo.upper() != "DQN":
                    env = DummyVecEnv([lambda: env])

                model.set_env(env)
                obs = env.reset()
                done = False
                index = 0
                test_cases_vector_prob = []

                for index in range(0, len(test_cases)):
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, done, info = env.step(action)
                    test_cases_vector_prob.append(
                        {"index": index, "prob": action})

                    if done:
                        assert len(test_cases) == index + 1, (
                            "Evaluation is finished without iterating all "
                            "test cases "
                        )
                        break

                test_cases_vector_prob = sorted(
                    test_cases_vector_prob, key=lambda x: x["prob"], reverse=False
                )  # the lower the rank, te higher the priority
                sorted_test_cases = []

                for test_case in test_cases_vector_prob:
                    sorted_test_cases.append(
                        test_cases[test_case["index"]])

                return sorted_test_cases

            elif mode.upper() == "LISTWISE":

                test_cases = env.cycle_logs.test_cases

                if algo.upper() != "DQN":
                    env = DummyVecEnv([lambda: env])

                model.set_env(env)
                obs = env.reset()
                done = False
                i = 0

                while True and i < 1000000:
                    i = i + 1
                    action, _states = model.predict(obs, deterministic=False)

                    if agent_actions.count(action) == 0 and action < len(test_cases):
                        if isinstance(action, list) or isinstance(action, np.ndarray):
                            if action.size == 1:
                                agent_actions.append(action)
                            else:
                                agent_actions.append(action[0])
                        else:
                            agent_actions.append(action)

                    obs, rewards, done, info = env.step(action)

                    if done:
                        break
                sorted_test_cases = []

                for index in agent_actions:
                    sorted_test_cases.append(test_cases[index])

                if i >= 1000000:
                    sorted_test_cases = test_cases

                return sorted_test_cases

            elif mode.upper() == "LISTWISE2":
                env = model.get_env()
                obs = env.reset()
                action, _states = model.predict(obs, deterministic=True)
                env.step(action)
                if algo.upper() != "DQN":
                    return env.get_attr("sorted_test_cases")[0]
                else:
                    return env.sorted_test_cases
    except RecursionError:
        # print in terminal in red
        print("\033[91m", "Recursion error encountered. Skipiing.", "\033[0m")
        pass
