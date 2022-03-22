from asyncio.log import logger
import sys, os, math, time, logging, random
import multiprocessing as mp
import numpy as np

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
from pbt.population import Population


population = Population(
    "POINTWISE", "simple", "data/iofrol-additional-features.csv", {}, "A2C", 1000, 1, 10
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
