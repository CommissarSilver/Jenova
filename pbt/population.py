from asyncio.log import logger
import sys, os, math, time, logging
import multiprocessing as mp

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from cv2 import log
from utils import ci_cycle, data_loader, utils
from agents import agent
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import update_learning_rate
from envs.PairWiseEnv import CIPairWiseEnv
from envs.PointWiseEnv import CIPointWiseEnv
from envs.CIListWiseEnvMultiAction import CIListWiseEnvMultiAction
from envs.CIListWiseEnv import CIListWiseEnv
from datetime import datetime
from errno import EADDRNOTAVAIL
from fileinput import filename
import logging


class Population:
    def __init__(
        self,
        environment_mode: str,
        dataset_type: str,
        train_data: str,
        hyper_parameters: dict,
        algorithm: str,
        episodes: int,
        population_id: str,
        number_of_agents: int,
    ) -> None:
        self.environment_mode = environment_mode
        self.dataset_type = dataset_type
        self.train_data = train_data
        self.hyper_parameters = hyper_parameters
        self.algorithm = algorithm
        self.episodes = episodes
        self.population_id = population_id
        self.number_of_agents = number_of_agents

        results_path = (
            f"./results/{self.algorithm}/{self.environment_mode}/{self.population_id}/"
        )
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        self.agents = [
            agent.Agent(
                self.environment_mode,
                self.dataset_type,
                self.train_data,
                self.hyper_parameters,
                self.algorithm,
                self.episodes,
                self.population_id,
                i,
                results_path,
            )
            for i in range(self.number_of_agents)
        ]

    def initialize_population(self, train=True):
        """
        initialize population of agents
        """
        for agent in self.agents:
            agent.initialize_agent()

        if train:
            processes = [
                mp.Process(
                    target=self.agents[i].train_agent,
                    args=(agent.environment, agent.environment_steps),
                )
                for i in range(len(self.agents))
            ]
            print("\033[91m" + "*" * 40 + "\033[0m")
            for process in processes:
                process.start()

            for process in processes:
                process.join()
            print("\033[91m" + "*" * 40 + "\033[0m")

            for agent in self.agents:
                agent.first_time = False

    def train_population(self):
        """
        train a population of agents
        """
        for agent in self.agents:
            agent.environment, agent.environment_steps = agent.get_environment()

        processes = [
            mp.Process(
                target=self.agents[i].train_agent,
                args=(agent.environment, agent.environment_steps),
            )
            for i in range(len(self.agents))
        ]
        print("\033[91m" + "*" * 40 + "\033[0m")
        for process in processes:
            process.start()

        for process in processes:
            process.join()
        print("\033[91m" + "*" * 40 + "\033[0m")


if __name__ == "__main__":
    "unit test"
    population = Population(
        "POINTWISE",
        "simple",
        "data/iofrol-additional-features.csv",
        {},
        "A2C",
        1000,
        1,
        10,
    )
    population.initialize_population()
    population.train_population()
    population.train_population()
    population.train_population()
    population.train_population()
    population.train_population()
    print("hello")
