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
		self.sc = []

		#TODO Build the rest of the subcritic system.
		# actor.layers = [state, output_layer1, output_layer2, output_layer3, ..., action = output_layer[-1]]
		
		# Make a PolynomialCritic of order ORDER for every neuron.
		#    Enumerate over every layer, l, in actor.layers[:-1]
		#    actor.layers[l] is the state for the polynomial critic
		#    for every neuron n in actor.layers[l+1], actor.layers[l+1][n] is
		#        the action for that polynomial critic.

		#   To make a polynomial critic, make a placeholder for the state of size actor.layers[l].shape
		#   and a placeholder for action of size actor.layers[l+1][n].shape ([1]).
		#   then let x =tf.concat(state_placeholder, action_placeholder) and the output of this polynomial
		#   critic will be Qn = x^TWx if order=2, or Qn = xW, if order =1, etc...

		pass


	def perceive(self, activations, reward, done):

		# In here do exactly the same method as ddpg for training its critic except do it 
		# for all critics.

		# Activations is the output of [state, output_layer1, output_layer2, output_layer3, ..., action = output_layer[-1]]
		
		# We need to make a replay buffer for the subcritics be a window of the last N time steps.
		# So do not use the replay buffer used for normal DDPG
		pass

	def q(self, activations):
		# Return the target Q of every single subcritic for plotting :)
		pass