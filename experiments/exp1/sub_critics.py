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
        self.sess = ddpg_agent.sess
		self.actor = ddpg_agent.actor_network
		self.count = 1
		self.critics = []

		#TODO Build the rest of the subcritic system.
		# actor.layers = [state, output_layer1, output_layer2, output_layer3, ..., action = output_layer[-1]]

        # Make a PolynomialCritic of order ORDER for every neuron.
        #    Enumerate over every layer, l, in actor.layers[:-1]
        for layer in self.actor.layers:
            # TODO fix this place holder
            neuron_count = layer.output_size
		#    for every neuron n in actor.layers[l+1], actor.layers[l+1][n] is
		#        the action for that polynomial critic.
            for _ in range(neuron_count):
                # TODO fix this is place holder
                #   and a placeholder for action of size actor.layers[l+1][n].shape ([1]).
                state_dim = layer.input_size
                action_dim = 1
                self.critics.append(PolynomialCritic(self.sess, state_dim, action_dim, order ))


		#    actor.layers[l] is the state for the polynomial critic


		#   To make a polynomial critic, make a placeholder for the state of size actor.layers[l].shape



		pass


	def perceive(self, activations, reward, done):

		# In here do exactly the same method as ddpg for training its critic except do it
		# for all critics.

		# Activations is the output of [state, output_layer1, output_layer2, output_layer3, ..., action = output_layer[-1]]

		# We need to make a replay buffer for the subcritics be a window of the last N time steps.
		# So do not use the replay buffer used for normal DDPG
		pass

	def q(self, activations):
		"""
        Return the Q of every single subcritic for plotting
        """
        # alternative if we need target output in addtion to q_value 
        # return [(sc.q_value_output, sc.target_q_value_output) for sc in self.critics]
		return [sc.q_value_output for sc in self.critics]
