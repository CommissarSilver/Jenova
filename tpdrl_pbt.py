import sys, os, math, time, logging, random, argparse
import multiprocessing as mp

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from agents import agent
import logging
from pbt.population import Population

parser = argparse.ArgumentParser(description="Jenova")
sys.setrecursionlimit(1000000)
print("Jenova was a calamity that fell from the sky a long, long time ago.")
print("Also reise, reise mein Bruder (๑•̀ㅂ•́)و")

parser.add_argument(
    "-m",
    "--environment_mode",
    help="Environment's mode. Either pairwise, pointwise, or listwise",
    required=False,
    default="pointwise",
)
parser.add_argument(
    "-d",
    "--dataset_type",
    help="Dataset type. Either simple or enriched",
    required=False,
    default="simple",
)
parser.add_argument(
    "-t",
    "--train_data",
    help="Path to train set file",
    required=False,
    default="data/iofrol-additional-features.csv",
)
parser.add_argument(
    "-a",
    "--algorithm",
    help="Algorithm for Angets. Either DQN, A2C, or PPO",
    required=False,
    default="A2C",
)
parser.add_argument(
    "-e",
    "--episodes",
    help="Number of episodes for training the agents on environment",
    required=False,
    default="200",
)
parser.add_argument(
    "-p", "--population_id", help="ID of the population", required=False, default="1"
)
parser.add_argument(
    "-na",
    "--number_of_agents",
    help="Number of agents in the population",
    required=False,
    default=mp.cpu_count(),
)
args = parser.parse_args()

population = Population(
    environment_mode=args.environment_mode.upper(),
    dataset_type=args.dataset_type,
    train_data=args.train_data,
    hyper_parameters={},
    algorithm=args.algorithm,
    episodes=int(args.episodes),
    population_id=int(args.population_id),
    number_of_agents=10,
)
population.initialize_population(train=False)
train_cycles = population.agents[0].reportDatasetInfo(
    population.agents[0].test_case_data
)
population.train_population()
for agent in population.agents:
    agent.first_time = False
for train_cycles in range(1, train_cycles - 1):
    population.train_population()
    population.train_population()
    population.train_population()
    population.train_population()
    population.train_population()
    population.train_population()
    population.train_population()
    population.train_population()
    population.train_population()
    population.train_population()
    population.train_population()

    # population.sort_population()
    # population.exploit()
    # population.explore()

