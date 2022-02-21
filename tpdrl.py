from errno import EADDRNOTAVAIL
from utils import ci_cycle, data_loader, utils
from stable_baselines3.common.monitor import Monitor
from envs.PairWiseEnv import CIPairWiseEnv
from envs.PointWiseEnv import CIPointWiseEnv
from envs.CIListWiseEnvMultiAction import CIListWiseEnvMultiAction
from envs.CIListWiseEnv import CIListWiseEnv
import math
import time


class Config:
    def __init__(self):
        self.padding_digit = -1
        self.win_size = -1
        self.dataset_type = "simple"
        self.max_test_cases_count = 400
        self.training_steps = 10000
        self.discount_factor = 0.9
        self.experience_replay = False
        self.first_cycle = 1
        self.cycle_count = 100
        self.train_data = "../data/tc_data_paintcontrol.csv"
        self.output_path = "../data/DQNAgent"
        self.log_file = "log.csv"


# find the cycle with maximum number of test cases
def get_max_test_cases_count(cycle_logs: list):
    # psuedo code:
    # for each cycle
    #   get the number of test cases
    #   if the number is greater than the max_test_cases_count
    #       set the max_test_cases_count to the number
    # return the max_test_cases_count
    max_test_cases_count = 0
    for cycle_log in cycle_logs:
        if cycle_log.get_test_cases_count() > max_test_cases_count:
            max_test_cases_count = cycle_log.get_test_cases_count()

    return max_test_cases_count


def reportDatasetInfo(test_case_data: list):
    cycle_cnt = 0
    failed_test_case_cnt = 0
    test_case_cnt = 0
    failed_cycle = 0
    for cycle in test_case_data:
        if cycle.get_test_cases_count() > 5:
            cycle_cnt = cycle_cnt + 1
            test_case_cnt = test_case_cnt + cycle.get_test_cases_count()
            failed_test_case_cnt = (
                failed_test_case_cnt + cycle.get_failed_test_cases_count()
            )
            if cycle.get_failed_test_cases_count() > 0:
                failed_cycle = failed_cycle + 1

    print("Test Case info:")
    print(f"    \033[91m Number of cycles: {str(cycle_cnt)} \033[0m")
    print(f"    \033[91m Number of total test cases: {str(test_case_cnt)} \033[0m")
    print(f"    \033[91m Number of failed cycles: {str(failed_cycle)} \033[0m")
    print(
        f"    \033[91m Number of failed test cases: {str(failed_test_case_cnt)} \033[0m"
    )
    print(
        f"    \033[91m Failure rate: {str(round(failed_test_case_cnt/test_case_cnt,2)*100)} \033[0m"
    )


def run_experiment(
    test_case_data,
    env_mode,
    episodes,
    start_cycle,
    verbos,
    end_cycle,
    dataset_name,
    conf,
):
    # TODO - add logging
    # TODO - add saving of model (DONE)
    # TODO - add loding of previous model (DONE)
    # TODO - add logging training info
    # TODO - When will this endless, useless, fruitless torture end? Am I in this earth just to suffer? One must imagine sysyphus happy!
    # TODO - These need to go into a for loop. for each cycle train and tst buddy.
    start_cycle = 0
    end_cycle = len(test_case_data)
    first_time = True
    # generate a stirng with todays date

    model_save_path = f"./models/{time.strftime('%Y-%m-%d')}_DQN_{env_mode}"
    for i in range(start_cycle, end_cycle - 1):
        if env_mode.upper() == "Pointwise".upper():
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N, 2) + 1)))
            env = CIPointWiseEnv(test_case_data[i], conf)

        elif env_mode.upper() == "Pairwise".upper():
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N, 2) + 1)))
            env = CIPointWiseEnv(test_case_data[i], conf)

        elif env_mode.upper() == "Listwise".upper():
            conf.max_test_cases_count = get_max_test_cases_count(test_case_data)
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N, 2) + 1)))
            env = CIListWiseEnv(test_case_data[i], conf)

        elif env_mode.upper() == "ListwiseMultiAction".upper():
            conf.max_test_cases_count = get_max_test_cases_count(test_case_data)
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N, 2) + 1)))
            env = CIListWiseEnvMultiAction(test_case_data[i], conf)

        print(
            "\033[92m Training agent with replaying of cycle: "
            + str(i)
            + " with steps "
            + str(steps)
            + " \033[0m"
        )
        env = Monitor(env)
        if first_time:
            agent = utils.create_model("DQN", env)
            agent.learn(total_timesteps=100)
            agent.save(model_save_path)
            first_time = False
        else:
            agent = utils.load_model("DQN", env, model_save_path)


# TODO: Find out what these configs are for
conf = Config()
conf.win_size = 10
conf.first_cycle = 0
conf.cycle_count = 9999999
conf.output_path = (
    "../experiments/"
    + "simple"
    + "/"
    + "DQN"
    + "/"
    + "test"
    + "_"
    + str(conf.win_size)
    + "/"
)
conf.log_file = (
    conf.output_path
    + "simple"
    + "_"
    + "DQN"
    + "_"
    + "test"
    + "_"
    + "100"
    + "_"
    + str(conf.win_size)
    + "_log.txt"
)
conf.dataset_type = "simple"
conf.train_data = "data/iofrol-additional-features.csv"

test_data_loader = data_loader.TestCaseExecutionDataLoader(
    "data/iofrol-additional-features.csv", "simple"
)
test_data = test_data_loader.load_data()
ci_cycle_logs = test_data_loader.pre_process()
reportDatasetInfo(test_case_data=ci_cycle_logs)
run_experiment(ci_cycle_logs, "listwise".upper(), 1000, 0, False, 12000, "", conf)

