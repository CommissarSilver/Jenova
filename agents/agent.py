from asyncio.log import logger
import sys, os, math, time, logging, random
import multiprocessing as mp

from numpy import block

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from utils import ci_cycle, data_loader, utils
from stable_baselines3.common.monitor import Monitor
from envs.PairWiseEnv import CIPairWiseEnv
from envs.PointWiseEnv import CIPointWiseEnv
from envs.CIListWiseEnvMultiAction import CIListWiseEnvMultiAction
from envs.CIListWiseEnv import CIListWiseEnv
from datetime import datetime

import logging
from stable_baselines3.common.utils import update_learning_rate


class Config:
    """
    Instead of a bunch of parameters, this class is used to store all the parameters that are used in the experiment.
    """

    def __init__(self):
        """
        Constructor of the Config class. YOU DON'T SAY ʘ‿ʘ !
        """
        self.padding_digit = -1  # don't know what this is for
        self.win_size = 10  # don't know what this is for
        self.dataset_type = None  # either simple or enriched
        self.max_test_cases_count = 400
        self.training_steps = 10000
        self.discount_factor = 0.9
        self.experience_replay = False
        self.first_cycle = 1  # delete this?
        self.cycle_count = 9999999  # what is this for?
        self.train_data = None


class Agent:
    """
    Each agent is an "individual" in the population.
    Each agent is responsible for its own training and testing.
    """

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
        experiment_results_dir: str,
    ) -> None:
        """
        Constructor for the agent class.
        

        Args:
            environment_mode (str): either "pairwise" or "pointwise" or "listwise"
            dataset_type (str): either "simple" or "enriched"
            train_data (str): path to the training data
            hyper_parameters (dict): **this is important** a dictionary that contains the hyperparameters to set for each algorithm.
                                        DEPENDING ON THE ALGORITHM THIS DICTIONARY WILL HAVE DIFFERENT KEYS.
            algorithm (str): either "DQN" or "A2C" or "PPO"
            episodes (int): idk tbh
            population_id (str): to keep track of different experiments
            id (int): each agent is differntiated by an id
        """

        self.environemnt_mode = environment_mode
        self.hyper_parameters = hyper_parameters
        self.algorithm = algorithm
        self.test_case_data = None
        self.experiment_results_dir = experiment_results_dir
        self.conf = Config()
        self.conf.dataset_type = dataset_type
        self.conf.train_data = train_data
        self.episodes = episodes
        self.cycle_num = 0
        self.id = id
        self.population_id = population_id
        self.first_time = True

        logging.basicConfig(
            filename=f"runlog.log",
            filemode="a",
            format="%(name)s - %(module)s - %(funcName)s - %(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.logger = logging.getLogger(__name__)
        # Metrics for agent's performance
        self.apfds = []  # !!! - Average Percentage of Faults Detected
        self.nrpas = []  # !!! - Normalized Rank Percentile Average

    def get_max_test_cases_count(self, cycle_logs: list) -> int:
        """
        find the max number of test cases in all the training data

        Args:
            cycle_logs (list): training data

        Returns:
            int: maximum number of test cases in all the training data
        """
        # psuedo code:
        # for each cycle
        #   get the number of test cases
        #   if the number is greater than the max_test_cases_count
        #       set the max_test_cases_count to the number
        # return the max_test_cases_count
        try:
            max_test_cases_count = 0
            for cycle_log in cycle_logs:
                if cycle_log.get_test_cases_count() > max_test_cases_count:
                    max_test_cases_count = cycle_log.get_test_cases_count()

            return max_test_cases_count
        except Exception as e:
            self.logger.exception("Exception in get_max_test_cases_count")
            sys.exit(1)

    def reportDatasetInfo(self, test_case_data: list, print_info=False) -> None:
        """
        Print out the training data information

        Args:
            test_case_data (list): test case data
        """
        try:
            cycle_cnt = 0
            failed_test_case_cnt = 0
            test_case_cnt = 0
            failed_cycle = 0
            for cycle in test_case_data:
                if cycle.get_test_cases_count() > 5:
                    cycle_cnt = cycle_cnt + 1
                    test_case_cnt = test_case_cnt + cycle.get_test_cases_count()
                    failed_test_case_cnt = failed_test_case_cnt + cycle.get_failed_test_cases_count()

                    if cycle.get_failed_test_cases_count() > 0:
                        failed_cycle = failed_cycle + 1
            if print_info:
                print("\033[34mN Test Case info:\033[0m")
                print(f"    \033[91m Number of cycles: {str(cycle_cnt)} \033[0m")
                print(f"    \033[91m Number of total test cases: {str(test_case_cnt)} \033[0m")
                print(f"    \033[91m Number of failed cycles: {str(failed_cycle)} \033[0m")
                print(f"    \033[91m Number of failed test cases: {str(failed_test_case_cnt)} \033[0m")
                print(f"    \033[91m Failure rate: {str(round(failed_test_case_cnt/test_case_cnt,2)*100)} \033[0m")
            return len(test_case_data)
        except Exception as e:
            self.logger.exception("Exception in reportDatasetInfo")
            sys.exit(1)

    def get_environment(self) -> tuple:
        """
        create the environment for the test case as a gym environment so that the agent can interact with it.

        Returns:
            tuple: a tuple containing the environment and the number of steps in the environment
        """
        try:
            while True:
                if (self.test_case_data[self.cycle_num].get_test_cases_count() < 6) or (
                    (self.conf.dataset_type == "simple")
                    and (self.test_case_data[self.cycle_num].get_failed_test_cases_count() < 1)
                ):
                    self.cycle_num += 1
                    self.logger.info(
                        f"Agent {self.id} - Cycle {self.cycle_num} is not enough test cases to run the test"
                    )
                    continue

                if self.environemnt_mode.upper() == "POINTWISE":
                    N = self.test_case_data[self.cycle_num].get_test_cases_count()
                    steps = int(self.episodes * (N * (math.log(N, 2) + 1)))
                    env = CIPointWiseEnv(self.test_case_data[self.cycle_num], self.conf)
                    break

                elif self.environemnt_mode.upper() == "PAIRWISE".upper():
                    N = self.test_case_data[self.cycle_num].get_test_cases_count()
                    steps = int(self.episodes * (N * (math.log(N, 2) + 1)))
                    env = CIPairWiseEnv(self.test_case_data[self.cycle_num], self.conf)
                    break

                elif self.environemnt_mode.upper() == "LISTWISE".upper():
                    self.conf.max_test_cases_count = self.get_max_test_cases_count(self.test_case_data)
                    N = self.test_case_data[self.cycle_num].get_test_cases_count()
                    steps = int(self.episodes * (N * (math.log(N, 2) + 1)))
                    env = CIListWiseEnv(self.test_case_data[self.cycle_num], self.conf)
                    break

                elif self.environemnt_mode.upper() == "LISTWISEMULTIACTION".upper():
                    self.conf.max_test_cases_count = self.get_max_test_cases_count(self.test_case_data)
                    N = self.test_case_data[self.cycle_num].get_test_cases_count()
                    steps = int(self.episodes * (N * (math.log(N, 2) + 1)))
                    env = CIListWiseEnvMultiAction(self.test_case_data[self.cycle_num], self.conf)
                    break

            self.cycle_num += 1

            return Monitor(env), steps

        except Exception as e:
            self.logger.exception("Exception in get_environment")
            sys.exit(1)

    def initialize_agent(self) -> None:
        """
        This function should be called once before the agent is trained.
        It sets the agent up for training by setting a model save path, setting the agent's parameters and creating its model.
        """
        try:
            # if the directory for saving the model does not exist, create it
            save_path = f"./models/{self.algorithm}/{self.environemnt_mode}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.model_save_path = save_path + f"/P{self.population_id}_A{self.id}"

            # should only be set up once
            test_data_loader = data_loader.TestCaseExecutionDataLoader(self.conf.train_data, self.conf.dataset_type)
            test_data = test_data_loader.load_data()
            ci_cycle_logs = test_data_loader.pre_process()
            self.test_case_data = ci_cycle_logs
            self.end_cycle = len(self.test_case_data)

            self.logger.info(f"Agent {self.id} initialized")
        except Exception as e:
            self.logger.exception("Exception in inialize_agent")
            sys.exit(1)

    def train_agent(self, environment, environment_steps, test_results, pbt_info=None) -> None:
        """
        If this is the first time the agent is being trained, it will create a new model, trains it, and save it.
        If not, it will load the model from the save path and train it.
        """
        try:
            if self.first_time:  # if it's agent's first time, a model should be created
                self.model = utils.create_model(self.algorithm, environment, self.hyper_parameters)
            else:  # if it's not agent's first time, a model should be loaded
                self.model = utils.load_model(self.algorithm, environment, self.model_save_path)

            print(
                "\033[92m"
                + "Agent "
                + "\033[93m"
                + f"{self.id}"
                + "\033[0m \033[92m"
                + "training on cycle "
                + "\033[93m"
                + f"{self.cycle_num} "
                + "\033[92m"
                + "with "
                + "\033[93m"
                + f"{environment_steps} "
                + "\033[0m \033[92m"
                + "steps"
                + " \033[0m"
            )

            if pbt_info:  # if agent has been selected to go through PBT operation
                try:
                    # get replacement information from the queue
                    info = pbt_info.get(block=False)
                    print(f'Agent {self.model_save_path} is being replaced with {info["replacement_model_save_path"]}')
                    # load the replacement model
                    self.model = utils.load_model(self.algorithm, environment, info["replacement_model_save_path"])
                    # load the replacement model's hyper parameters
                    self.hyper_parameters = info["replacement_hyperparameters"]
                    # perturb the learning rate
                    update_learning_rate(
                        self.model.policy.optimizer,
                        learning_rate=self.hyper_parameters["learning_rate"] * random.uniform(0.8, 1.2),
                    )

                except Exception as e:  # if and exception has happened here, just load agent's previous model
                    logger.exception(f"agent {self.id} problem with PBT operation")
                    self.model = utils.load_model(self.algorithm, environment, self.model_save_path)

            try:  # train the model
                self.model.learn(total_timesteps=environment_steps)

            except Exception as e:  # if an exception happens during training, the agent is considered dead
                print("Agent has died")
                logger.critical(f"Agent {self.id} is dead")
                # put "agent mat" in queue to mark the agent as dead
                test_results.put({"agent_id": self.id, "apfd": "agent mat", "nrpa": "agent mat"})
                # terminate training
                return

            # save agent's model
            self.model.save(self.model_save_path)
            # test the agent
            self.test_agent()
            # put test results in the queue for PBT operations
            test_results.put({"agent_id": self.id, "apfd": self.apfds[-1], "nrpa": self.nrpas[-1]})

            print(
                "\033[92m"
                + "Agent "
                + "\033[93m"
                + f"{self.id}"
                + "\033[0m \033[92m"
                + "trained on cycle "
                + "\033[93m"
                f"{self.cycle_num} "
                + "\033[92m "
                + "with "
                + "\033[93m"
                + f"{environment_steps} "
                + "\033[0m \033[92m"
                + "steps "
                + +" \033[0m"
            )

            self.logger.info(f"Agent {self.id} trained on {environment_steps} steps")

        except Exception as e:
            # too many things can cause this. Any exception is considered a fatal error and the agent must be terminated
            self.logger.exception("Exception in train_agent")
            return

    # TODO: #15 add propoer documentation for this
    def test_agent(self) -> None:
        """
        THIS IS A NO MAN'S LAND THAT I DON'T WANT TO APPROACH ┬┴┬┴┤(･_├┬┴┬┴
        """
        try:
            self.test_cycle_num = self.cycle_num + 1
            while (
                (self.test_case_data[self.test_cycle_num].get_test_cases_count() < 6)
                or (
                    (self.conf.dataset_type == "simple")
                    and (self.test_case_data[self.test_cycle_num].get_failed_test_cases_count() == 0)
                )
            ) and (self.test_cycle_num < self.end_cycle):
                self.test_cycle_num = self.test_cycle_num + 1

            if self.test_cycle_num >= self.end_cycle - 1:
                return
            if self.environemnt_mode.upper() == "PAIRWISE":
                env_test = CIPairWiseEnv(self.test_case_data[self.test_cycle_num], self.conf)
            elif self.environemnt_mode.upper() == "POINTWISE":
                env_test = CIPointWiseEnv(self.test_case_data[self.test_cycle_num], self.conf)
            elif self.environemnt_mode.upper() == "LISTWISE":
                env_test = CIListWiseEnv(self.test_case_data[self.test_cycle_num], self.conf)
            elif self.environemnt_mode.upper() == "LISTWISE2":
                env_test = CIListWiseEnvMultiAction(self.test_case_data[self.test_cycle_num], self.conf)

            test_case_vector, rewards_sum = utils.test_agent(
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
                apfd = self.test_case_data[self.test_cycle_num].calc_APFD_ordered_vector(test_case_vector)
                apfd_optimal = self.test_case_data[self.test_cycle_num].calc_optimal_APFD()
                apfd_random = self.test_case_data[self.test_cycle_num].calc_random_APFD()
                self.apfds.append(apfd)
            else:
                apfd = 0
                apfd_optimal = 0
                apfd_random = 0

            nrpa = self.test_case_data[self.test_cycle_num].calc_NRPA_vector(test_case_vector)
            self.nrpas.append(nrpa)
            print(
                f"\033[92mTesting agent \033[93m{self.id}\033[92m on cycle \033[93m"
                + str(self.test_cycle_num)
                + " \033[92mresulted in APFD: \033[93m"
                + str(apfd)
                + " \033[92m, NRPA: \033[93m"
                + str(nrpa)
                + " \033[92m, optimal APFD: \033[93m"
                + str(apfd_optimal)
                + " \033[92m, random APFD: \033[93m"
                + str(apfd_random)
                + " \033[92m, # failed test cases: \033[93m"
                + str(self.test_case_data[self.test_cycle_num].get_failed_test_cases_count())
                + " \033[92m, # test cases: \033[93m"
                + str(self.test_case_data[self.test_cycle_num].get_test_cases_count())
                + " \033[92m, Rewards sum: \033[93m"
                + str(rewards_sum)
                + "\033[0m",
                flush=True,
            )

            experiment_results = open(self.experiment_results_dir + f"{self.id}_{self.population_id}_results.csv", "a")
            experiment_results.write(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                + ","
                + f"{self.id}"
                + ","
                + self.environemnt_mode
                + ","
                + self.algorithm
                + ","
                + (self.model_save_path)
                + ","
                + str(self.episodes)
                + ","
                + str(self.environment_steps)
                + ","
                + str(cycle_id_text)
                + ","
                + str(self.test_case_data[self.test_cycle_num].get_test_cases_count())
                + ","
                + str(self.test_case_data[self.test_cycle_num].get_failed_test_cases_count())
                + ","
                + str(apfd)
                + ","
                + str(nrpa)
                + ","
                + str(apfd_random)
                + ","
                + str(apfd_optimal)
                + ","
                + str(self.hyper_parameters["learning_rate"])
                + ","
                + str(self.hyper_parameters["gamma"])
                + ","
                + str(rewards_sum[0])
                + "\n"
            )
            experiment_results.close()
        except Exception as e:
            self.logger.exception("Exception in test_agent")
            return


if __name__ == "__main__":
    """
    for unit testing
    """
    agent1 = Agent(
        "POINTWISE",
        "simple",
        "data/iofrol-additional-features.csv",
        {},
        "A2C",
        1000,
        1,
        1,
        f"./results/A2C/pointwise.csv",
    )
    agent2 = Agent(
        "POINTWISE",
        "simple",
        "data/iofrol-additional-features.csv",
        {},
        "A2C",
        1000,
        1,
        2,
        f"./results/A2C/pointwise.csv",
    )
    agent1.train_agent()
    agent1.test_agent()
    agent1.logger.info("Agent 1 trained")

    agent2.train_agent()
    agent2.test_agent()
    agent2.logger.info("Agent 2 trained")

