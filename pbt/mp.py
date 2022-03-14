import multiprocessing as mp
import tpdrl
from utils import utils
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


def test():

    conf = tpdrl.Config()
    conf.win_size = 10
    conf.first_cycle = 0
    conf.cycle_count = 9999999

    conf.dataset_type = "simple"
    conf.train_data = "data/iofrol-additional-features.csv"

    test_data_loader = data_loader.TestCaseExecutionDataLoader(
        "data/iofrol-additional-features.csv", "simple"
    )
    test_data = test_data_loader.load_data()
    ci_cycle_logs = test_data_loader.pre_process()
    tpdrl.reportDatasetInfo(test_case_data=ci_cycle_logs)

    tpdrl.run_experiment(
        ci_cycle_logs, "pointwise".upper(), 1000, 0, False, 12000, "", conf
    )


if __name__ == "__main__":
    # print the number of processors in blue
    print("\033[34mNumber of processors: {}\033[0m".format(mp.cpu_count()))

    agents = []

