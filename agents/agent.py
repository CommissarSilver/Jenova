from errno import EADDRNOTAVAIL
from fileinput import filename

from cv2 import log
from utils import ci_cycle, data_loader, utils
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import update_learning_rate
from envs.PairWiseEnv import CIPairWiseEnv
from envs.PointWiseEnv import CIPointWiseEnv
from envs.CIListWiseEnvMultiAction import CIListWiseEnvMultiAction
from envs.CIListWiseEnv import CIListWiseEnv
import math, time, os, logging
import sys
from datetime import datetime


class Config:
    """
    Instead of a bunch of parameters, this class is used to store all the parameters that are used in the experiment.
    """

    def __init__(self):
        """
        Constructor of the Config class. YOU DON'T SAY ʘ‿ʘ !
        """
        self.padding_digit = -1  # don't know what this is for
        self.win_size = -1  # don't know what this is for
        self.dataset_type = None  # either simple or enriched
        self.max_test_cases_count = 400
        self.training_steps = 10000
        self.discount_factor = 0.9
        self.experience_replay = False
        self.first_cycle = 1  # delete this?
        self.cycle_count = 100  # what is this for?
        self.train_data = None


class Agent:
    def __init__(
        self,
        environment_mode: str,
        dataset_type: str,
        train_data: str,
        hyper_parameters: dict,
        algorithm: str,
        episodes: int,
        population_id: str,
        id: int,
    ):
        self.environemnt_mode = environment_mode
        self.hyper_parameters = hyper_parameters
        self.algorithm = algorithm
        self.test_case_data = None
        self.logger = logging.getLogger(__name__)
        self.conf = Config()
        self.conf.dataset_type = dataset_type
        self.conf.train_data = train_data
        self.episodes = episodes
        self.cycle_num = 0
        self.id = id
        self.population_id = population_id
        self.first_time = True
        self.apfds = []  # !!! - Average Percentage of Faults Detected
        self.nrpas = []  # !!! - Normalized Rank Percentile Average

    def get_max_test_cases_count(self, cycle_logs: list):
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

    def reportDatasetInfo(self, test_case_data: list):
        # psudo code:
        # for each test case
        #
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

        print("\033[34mN Test Case info:\033[0m")
        print(f"    \033[91m Number of cycles: {str(cycle_cnt)} \033[0m")
        print(f"    \033[91m Number of total test cases: {str(test_case_cnt)} \033[0m")
        print(f"    \033[91m Number of failed cycles: {str(failed_cycle)} \033[0m")
        print(
            f"    \033[91m Number of failed test cases: {str(failed_test_case_cnt)} \033[0m"
        )
        print(
            f"    \033[91m Failure rate: {str(round(failed_test_case_cnt/test_case_cnt,2)*100)} \033[0m"
        )

    # TODO: #13 this is just ugly code - fix it later
    def get_environment(self):
        skip = False
        if (self.test_case_data[self.cycle_num].get_test_cases_count() < 6) or (
            (self.conf.dataset_type == "simple")
            and (self.test_case_data[self.cycle_num].get_failed_test_cases_count() < 1)
        ):
            skip = True

        if not skip:
            if self.environemnt_mode.upper() == "POINTWISE":
                N = self.test_case_data[self.cycle_num].get_test_cases_count()
                steps = int(self.episodes * (N * (math.log(N, 2) + 1)))
                env = CIPointWiseEnv(self.test_case_data[self.cycle_num], self.conf)

            elif self.environemnt_mode.upper() == "PAIRWISE".upper():
                N = self.test_case_data[self.cycle_num].get_test_cases_count()
                steps = int(self.episodes * (N * (math.log(N, 2) + 1)))
                env = CIPointWiseEnv(self.test_case_data[self.cycle_num], self.conf)

            elif self.environemnt_mode.upper() == "LISTWISE".upper():
                self.conf.max_test_cases_count = self.get_max_test_cases_count(
                    self.test_case_data
                )
                N = self.test_case_data[self.cycle_num].get_test_cases_count()
                steps = int(self.episodes * (N * (math.log(N, 2) + 1)))
                env = CIListWiseEnv(self.test_case_data[self.cycle_num], self.conf)

            elif self.environemnt_mode.upper() == "LISTWISEMULTIACTION".upper():
                self.conf.max_test_cases_count = self.get_max_test_cases_count(
                    self.test_case_data
                )
                N = self.test_case_data[self.cycle_num].get_test_cases_count()
                steps = int(self.episodes * (N * (math.log(N, 2) + 1)))
                env = CIListWiseEnvMultiAction(
                    self.test_case_data[self.cycle_num], self.conf
                )

        self.cycle_num += 1
        if not skip:
            return Monitor(env), steps

    def inialize_agent(self):
        save_path = f"./models/{self.algorithm}/{self.environemnt_mode}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model_save_path = save_path + f"/{self.population_id}_{self.id}"

        test_data_loader = data_loader.TestCaseExecutionDataLoader(
            self.conf.train_data, self.conf.dataset_type
        )
        test_data = test_data_loader.load_data()
        ci_cycle_logs = test_data_loader.pre_process()
        self.test_case_data = ci_cycle_logs
        self.end_cycle = len(self.test_case_data)

        environment = self.get_environment()

        self.model = utils.create_model(
            self.algorithm, environment, self.hyper_parameters
        )
        self.first_time = False

    def train_agent(self, steps: int):
        if self.first_time:
            self.inialize_agent()
            self.model.learn(total_timesteps=steps)
            self.model.save(self.model_save_path)
        else:
            environment = self.get_environment()
            self.model = utils.load_model(
                self.algorithm, environment, self.model_save_path
            )
            self.model.learn(steps)
            self.model.save(self.model_save_path)

    def test_agent(self):
        self.test_cycle_num = self.cycle_num + 1
        while (
            (self.test_case_data[self.test_cycle_num].get_test_cases_count() < 6)
            or (
                (self.conf.dataset_type == "simple")
                and (
                    self.test_case_data[
                        self.test_cycle_num
                    ].get_failed_test_cases_count()
                    == 0
                )
            )
        ) and (self.test_cycle_num < self.end_cycle):
            self.test_cycle_num = self.test_cycle_num + 1

        if self.test_cycle_num >= self.end_cycle - 1:
            pass
            # break
        if self.environemnt_mode.upper() == "PAIRWISE":
            env_test = CIPairWiseEnv(
                self.test_case_data[self.test_cycle_num], self.conf
            )
        elif self.environemnt_mode.upper() == "POINTWISE":
            env_test = CIPointWiseEnv(
                self.test_case_data[self.test_cycle_num], self.conf
            )
        elif self.environemnt_mode.upper() == "LISTWISE":
            env_test = CIListWiseEnv(
                self.test_case_data[self.test_cycle_num], self.conf
            )
        elif self.environemnt_mode.upper() == "LISTWISE2":
            env_test = CIListWiseEnvMultiAction(
                self.test_case_data[self.test_cycle_num], self.conf
            )

        test_case_vector = utils.test_agent(
            environment=env_test,
            algo=self.algorithm,
            model_path=self.model_save_path + ".zip",
            environment_mode=self.environemnt_mode.upper(),
        )
        test_case_id_vector = []

        for test_case in test_case_vector:
            test_case_id_vector.append(str(test_case["test_id"]))
            cycle_id_text = test_case["cycle_id"]

        if self.test_case_data[self.test_cycle_num].get_failed_test_cases_count() != 0:
            apfd = self.test_case_data[self.test_cycle_num].calc_APFD_ordered_vector(
                test_case_vector
            )
            apfd_optimal = self.test_case_data[self.test_cycle_num].calc_optimal_APFD()
            apfd_random = self.test_case_data[self.test_cycle_num].calc_random_APFD()
            self.apfds.append(apfd)
        else:
            apfd = 0
            apfd_optimal = 0
            apfd_random = 0

        nrpa = self.test_case_data[self.test_cycle_num].calc_NRPA_vector(
            test_case_vector
        )
        self.nrpas.append(nrpa)

