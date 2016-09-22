# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: William Guss
# Date: 2016.5.4
# -----------------------------------
import gym
import tensorflow as tf
import numpy as np
from ddpg import DDPG
from common.critic_network import CriticNetwork
from common.actor_network import ActorNetwork
from common.replay_buffer import ReplayBuffer


class SubCritics:
	def __init__(self, ddpg_agent : DDPG, order=1):
		"""
		Initializes the subcritics based on a particular ddpg agent.
		ORDER is currently unused.
		"""
		self.agent = ddpg_agent
		self.actor = ddpg_agent.actor_network
		self.count = 1

		#TODO Build the rest of the subcritic system.
		pass


	def perceive(self, activations, reward, done):
		pass

	def q(self, activations):
		pass