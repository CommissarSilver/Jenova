import sys, os, math, time, logging, random, argparse
import multiprocessing as mp

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from agents import agent as agent_module
import logging
from pbt.population import Population

parser = argparse.ArgumentParser(description="Jenova")
sys.setrecursionlimit(1000000)
print("Jenova was a calamity that fell from the sky a long, long time ago.")


parser.add_argument(
    "-m",
    "--environment_mode",
    help="Environment's mode. Either pairwise, pointwise, or listwise",
    required=False,
    default="listwise",
)
parser.add_argument(
    "-d", "--dataset_type", help="Dataset type. Either simple or enriched", required=False, default="simple"
)
parser.add_argument(
    "-t",
    "--train_data",
    help="Path to train set file",
    required=False,
    default="data/paintcontrol-additional-features.csv",
)
parser.add_argument(
    "-a", "--algorithm", help="Algorithm for Angets. Either DQN, A2C, or PPO", required=False, default="A2C"
)
parser.add_argument(
    "-e", "--episodes", help="Number of episodes for training the agents on environment", required=False, default="200"
)
parser.add_argument("-p", "--population_id", help="ID of the population", required=False, default="1")
parser.add_argument("-na", "--number_of_agents", help="Number of agents in the population", required=False, default=5)
args = parser.parse_args()

# TODO: initilizing the population like this gives the same hyperparameters for all agents. this is problem that needs fixing.
population = Population(
    environment_mode=args.environment_mode.upper(),
    dataset_type=args.dataset_type,
    train_data=args.train_data,
    hyper_parameters={"gamma": random.uniform(0, 1), "learning_rate": random.uniform(0.1, 0.0001)},
    algorithm=args.algorithm,
    episodes=int(args.episodes),
    population_id=int(args.population_id),
    number_of_agents=mp.cpu_count(),
)


population.initialize_population()
train_cycles = population.agents[0].reportDatasetInfo(population.agents[0].test_case_data)

for agent in population.agents:
    test_resulst = mp.Queue()
    agent.environment, agent.environment_steps = agent.get_environment()
    agent.train_agent(agent.environment, agent.environment_steps, test_resulst)
    agent.first_time = False


for train_cycles in range(1, train_cycles - 1):
    for agent in population.agents:
        test_resulst = mp.Queue()
        agent.environment, agent.environment_steps = agent.get_environment()
        agent.train_agent(agent.environment, agent.environment_steps, test_resulst)

