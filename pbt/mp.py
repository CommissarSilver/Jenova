import multiprocessing as mp
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import tpdrl
from cv2 import log
from utils import ci_cycle, data_loader, utils
import math, time, logging

from datetime import datetime


def train(
    dataset_type: str = "simple",
    train_data: str = "data/iofrol-additional-features.csv",
    environment_mode: str = "pointwise",
    hyper_parameters: dict = {},
):

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


# TODO: #12 change the name of the script. this is supposed to be a proof of concept.
if __name__ == "__main__":
    # print the number of processors in blue
    print("\033[34mNumber of processors: {}\033[0m".format(mp.cpu_count()))
    p1 = mp.Process(target=train)
    p2 = mp.Process(target=train)
    p3 = mp.Process(target=train)
    p4 = mp.Process(target=train)
    print("first one")
    p1.start()
    print("second one")
    p2.start()
    p3.start()
    p4.start()
    agents = []

