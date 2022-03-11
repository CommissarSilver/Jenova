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

from datetime import datetime

logging.basicConfig(
    filename=f'{time.strftime("%Y-%m-%d_%H-%M")}.log',
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)

# TODO - create a proper docstring for this class
class Config:
    """
    Instead of a bunch of parameters, this class is used to store all the parameters that are used in the experiment.
    """

    def __init__(self):
        """
        Constructor of the Config class.
        """
        self.padding_digit = -1  # don't know what this is for
        self.win_size = -1  # don't know what this is for
        self.dataset_type = "simple"  # either simple or enriched
        self.max_test_cases_count = 400
        self.training_steps = 10000
        self.discount_factor = 0.9
        self.experience_replay = (
            False  # TODO - remove this since it is not supported anymore
        )
        self.first_cycle = 1  # delete this?
        self.cycle_count = 100  # what is this for?
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


# print information regarding the test cases
def reportDatasetInfo(test_case_data: list):
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


# run experiment on test case data


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
    """
    This is where the agent gets trained and tested.
    If the agent has not been trained before, a new model is created, otherwise a previous model is loaded.   

    Args:
        test_case_data (_type_): _description_
        env_mode (_type_): _description_
        episodes (_type_): _description_
        start_cycle (_type_): _description_
        verbos (_type_): _description_
        end_cycle (_type_): _description_
        dataset_name (_type_): _description_
        conf (_type_): _description_
    """
    # TODO - add logging (DONE)
    # TODO - add saving of model (DONE)
    # TODO - add loding of previous model (DONE)
    # TODO - Cuatom callback
    # TODO - add logging training info (DONE)
    # TODO - When will this endless, useless, fruitless torture end? Am I in this earth just to suffer? One must imagine sisyphus happy!
    # TODO - These need to go into a for loop. for each cycle train and tst buddy. (DONE)
    # TODO - what is afpd and nrpa?
    logging.info(f"Starting experiment on {env_mode}/{algorithm} on {conf.train_data}")

    start_cycle = 0
    end_cycle = len(test_case_data)
    first_time = True
    algorithm = "A2C"  # TODO - add algorithm as a parameter

    # if the directory for saving results doesn't exit, create it.
    try:
        results_path = f"./results/{algorithm}/{env_mode}"
        if not os.path.exists(results_path):
            os.makedirs(results_path)
            logging.info("Results directory created")
        experiment_results = (
            f'{algorithm}_{env_mode}_{time.strftime("%Y-%m-%d_%H-%M")}.csv'
        )
        experiment_results = open(results_path + "/" + experiment_results, "w")
        experiment_results.write(
            "Timestamp,Mode,Algorithm,Model_Name,Episodes,Steps,Cycle_ID,Test_Cases,Failed_Test_Cases,APFD,NRPA,Random_APFD,Optimal_APFD\n"
        )
        logging.info("Results file created")
    except Exception as e:
        logging.critical(
            "Error while creating results directory or file", exc_info=True
        )

    # if the directory for saving models doesn't exits, create it
    try:
        save_path = f"./models/{algorithm}/{env_mode}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            model_save_path = save_path + f'/{time.strftime("%Y-%m-%d_%H-%M")}'
            logging.info("Model directory created")
        else:
            model_save_path = save_path + f'/{time.strftime("%Y-%m-%d_%H-%M")}'
            logging.info("Model directory exists")
    except Exception as e:
        logging.critical("Error while creating model directory", excet_info=True)

    apfds = []  # !!! - Average Percentage of Faults Detected
    nrpas = []  # !!! - Normalized Rank Percentile Average

    # as of now, there are 209 cycles. for each cycle, we need to create a separate environment.
    # then we need to train the agent on the environment.
    for i in range(start_cycle, end_cycle - 1):
        # build the environemnt for the current cycle
        if env_mode.upper() == "POINTWISE":
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N, 2) + 1)))
            env = CIPointWiseEnv(test_case_data[i], conf)

        elif env_mode.upper() == "PAIRWISE".upper():
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N, 2) + 1)))
            env = CIPointWiseEnv(test_case_data[i], conf)

        elif env_mode.upper() == "LISTWISE".upper():
            conf.max_test_cases_count = get_max_test_cases_count(test_case_data)
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N, 2) + 1)))
            env = CIListWiseEnv(test_case_data[i], conf)

        elif env_mode.upper() == "LISTWISEMULTIACTION".upper():
            conf.max_test_cases_count = get_max_test_cases_count(test_case_data)
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N, 2) + 1)))
            env = CIListWiseEnvMultiAction(test_case_data[i], conf)

        logging.info(f"Training agent on {env_mode}")

        print(
            "\033[92m Training agent with replaying of cycle: "
            + str(i)
            + " with "
            + str(steps)
            + " steps"
            + " \033[0m"
        )
        try:
            env = Monitor(env)
        except Exception as e:
            logging.critical(
                f"Error while creating monitor for {env_mode}", exc_info=True
            )

        if first_time:  # if a model doesn't esit, create a new one
            try:
                # create an agent with the given algorithm and environment
                agent = utils.create_model("A2C", env)
                # train the agent
                agent.learn(total_timesteps=steps)

                # ! THIS IS WHERE WE CAN UPDATE THE AGENT'S LEARNING RATE
                # update_learning_rate(agent.policy.optimizer, learning_rate=0.0001)

                # save agent's model
                agent.save(model_save_path)
                first_time = False
                logging.info("Agent trained successfully for first round")
            except Exception as e:
                logging.critical(
                    f"Error while training {algorithm} agent on for first time on {env_mode}",
                    exc_info=True,
                )
        else:  # if model exists, load it
            try:
                # load the agent with the given algorithm and environemnt and model path
                agent = utils.load_model("A2C", env, model_save_path)
                logging.info("Agent loaded successfully")
            except Exception as e:
                logging.critical(
                    f"Error while loading {algorithm} agent on {env_mode}",
                    exc_info=True,
                )

        j = i + 1  # test trained agent on next cycles
        while (
            (test_case_data[j].get_test_cases_count() < 6)
            or (
                (conf.dataset_type == "simple")
                and (test_case_data[j].get_failed_test_cases_count() == 0)
            )
        ) and (j < end_cycle):

            j = j + 1
        if j >= end_cycle - 1:
            break
        # after training, testing begins.
        # the environemnts' types are the same as the training envs.
        try:
            if env_mode.upper() == "PAIRWISE":
                env_test = CIPairWiseEnv(test_case_data[j], conf)
            elif env_mode.upper() == "POINTWISE":
                env_test = CIPointWiseEnv(test_case_data[j], conf)
            elif env_mode.upper() == "LISTWISE":
                env_test = CIListWiseEnv(test_case_data[j], conf)
            elif env_mode.upper() == "LISTWISE2":
                env_test = CIListWiseEnvMultiAction(test_case_data[j], conf)
            logging.info("Test environment created successfully")
        except Exception as e:
            logging.critical(f"Error while creating test environment", exc_info=True)

        test_time_start = datetime.now()
        # TODO - change algo to algorithm in the funciton parameters
        try:
            test_case_vector = utils.test_agent(
                env=env_test,
                algo=algorithm,
                model_path=model_save_path + ".zip",
                mode=env_mode.upper(),
            )
            logging.info("Test agent loaded successfuly")
        except Exception as e:
            logging.critical("Error while loading test agent", exc_info=True)

        test_time_end = datetime.now()
        test_case_id_vector = []

        for test_case in test_case_vector:
            test_case_id_vector.append(str(test_case["test_id"]))
            cycle_id_text = test_case["cycle_id"]

        try:
            if test_case_data[j].get_failed_test_cases_count() != 0:
                apfd = test_case_data[j].calc_APFD_ordered_vector(test_case_vector)
                apfd_optimal = test_case_data[j].calc_optimal_APFD()
                apfd_random = test_case_data[j].calc_random_APFD()
                apfds.append(apfd)
            else:
                apfd = 0
                apfd_optimal = 0
                apfd_random = 0

            nrpa = test_case_data[j].calc_NRPA_vector(test_case_vector)
            nrpas.append(nrpa)
            # test_time = millis_interval(test_time_start, test_time_end)
            # training_time = millis_interval(training_start_time, training_end_time)
            print(
                "Testing agent  on cycle "
                + str(j)
                + " resulted in APFD: "
                + str(apfd)
                + " , NRPA: "
                + str(nrpa)
                + " , optimal APFD: "
                + str(apfd_optimal)
                + " , random APFD: "
                + str(apfd_random)
                + " , # failed test cases: "
                + str(test_case_data[j].get_failed_test_cases_count())
                + " , # test cases: "
                + str(test_case_data[j].get_test_cases_count()),
                flush=True,
            )
            experiment_results.write(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                + ","
                + env_mode
                + ","
                + algorithm
                + ","
                + (model_save_path)
                + ","
                + str(episodes)
                + ","
                + str(steps)
                + ","
                + str(cycle_id_text)
                + ","
                + str(test_case_data[j].get_test_cases_count())
                + ","
                + str(test_case_data[j].get_failed_test_cases_count())
                + ","
                + str(apfd)
                + ","
                + str(nrpa)
                + ","
                + str(apfd_random)
                + ","
                + str(apfd_optimal)
                + "\n"
            )
        except RecursionError:
            # print below in red color
            print("\033[91m RecursionError \033[0m")
            logging.critical(
                f"Recursion Error while calculating APFD/NRPA on test case {j}",
                exc_info=True,
            )
        else:
            logging.critical(
                f"Error while testing agent on test case {j}", exc_info=True
            )


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
run_experiment(ci_cycle_logs, "pointwise".upper(), 1000, 0, False, 12000, "", conf)

