# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: William Guss
# Date: 2016.5.4
# -----------------------------------
import gym
import tensorflow as tf
import numpy as np
from ou_noise import OUNoise
from ddpg import DDPG
from common.critic_network import CriticNetwork
from common.actor_network import ActorNetwork
from common.replay_buffer import ReplayBuffer


class SubCritics:
	def __init__(self, ddpg_agent : DDPG, )