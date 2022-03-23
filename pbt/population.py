from asyncio.log import logger
import sys, os, math, time, logging, random
import multiprocessing as mp
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], ".."))

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

    def initialize_population(self, train: bool = True) -> None:
        """
        Initilizes the population of agents.

        Args:
            train (bool, optional): whether to train the agents after initialization  or not. Defaults to True.
        """
        for agent in self.agents:
            agent.initialize_agent()

    def train_population(self, test: str = True) -> None:
        """
        train the population of agents simultaneously
        ! Important, because of the mess that is python multiprocessing, keep test to true if you want to have some results to compare.
        Args:
            test (str, optional): whether to test the agent after training. Defaults to True.
        """
        test_results = mp.Queue()

        for agent in self.agents:
            agent.environment, agent.environment_steps = agent.get_environment()

        processes = [
            mp.Process(
                target=self.agents[i].train_agent,
                args=(agent.environment, agent.environment_steps, test_results),
            )
            for i in range(len(self.agents))
        ]
        print("\033[91m" + "*" * 40 + "\033[0m")
        for process in processes:
            process.start()

        for process in processes:
            process.join()

        for i in range(self.number_of_agents):
            agent_results = test_results.get(block=False)

            for agent in self.agents:
                if agent.id == agent_results["agent_id"]:
                    agent.apfds.append(agent_results["apfd"])
                    agent.nrpas.append(agent_results["nrpa"])

        print("\033[91m" + "*" * 40 + "\033[0m")

    def sort_population(self, sorting_criteria: str = "apfd") -> None:
        """
        sorts the agents from best to worst according to the sorting criteria.
        "APFD" stands for Average Percentage of Faults Detected. (higher is better)
        "NRPA" stands for Normalized Rank Percentile Average. (higher is better)
        Generally, "APFD" is preferred.
        
        Args:
            sorting_criteria (str, optional): either "apfd" or "nrpa". Defaults to "apfd".
        """
        if sorting_criteria == "apfd":
            self.agents.sort(key=lambda agent: np.average(agent.apfds), reverse=True)
        elif sorting_criteria == "nrpa":
            self.agents.sort(key=lambda agent: np.average(agent.nrpas), reverse=True)
        else:
            print("sorting criteria not supported")
            sys.exit(1)

    def explore(self) -> None:
        replacement_percentile = int(self.number_of_agents * 0.3)
        worst_agents = self.agents[-replacement_percentile:]

        for agent in worst_agents:
            agent.model = utils.load_model(
                agent.algorithm, agent.environment, agent.model_save_path
            )
            update_learning_rate(
                agent.model.policy.optimizer,
                learning_rate=agent.hyper_parameters["learning_rate"]
                * random.uniform(0.8, 1.2),
            )

    # TODO: #20 this function, causes the agents to hang because of loading and saving of the model. find a workaround.
    def exploit(self):
        replacement_percentile = int(self.number_of_agents * 0.3)
        worst_agents = self.agents[-replacement_percentile:]
        best_agents = self.agents[:replacement_percentile]

        for agent in worst_agents:
            chosen_replacement = random.choice(best_agents)
            agent.model = utils.load_model(
                agent.algorithm, agent.environment, chosen_replacement.model_save_path
            )
            agent.hyper_parameters = chosen_replacement.hyper_parameters
            agent.model.save(agent.model_save_path)
        print("done")


if __name__ == "__main__":
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
    population.initialize_population(train=False)
    population.train_population()
    for agent in population.agents:
        agent.first_time = False
    population.train_population()
    population.train_population()

    population.sort_population()
    population.exploit()
    print("hello")
