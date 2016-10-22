################################################################
# EXPERIMENT 2:
#
# In this experiment we will train an actor according to its subcritics
################################################################
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display
from itertools import chain
import gc
gc.enable()

from brain import DDPG
from brain import SubCritics
from brain.common.filter_env import makeFilteredEnv

def run_experiment(ENV_NAME='MountainCarContinuous-v0', EPISODES=10000, TEST=10):
    """
    Runs the experiment.
    """
    pass
if __name__ == '__main__':
    run_experiment()
