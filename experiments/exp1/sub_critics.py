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

from polynomial_critic import PolynomialCritic

class SubCritics:
    def __init__(self, ddpg_agent : DDPG, order=1, verbose=False):
        """
        Initializes the subcritics based on a particular ddpg agent.
        ORDER is currently unused.
        """
        self.agent = ddpg_agent
        self.sess = ddpg_agent.sess
        self.actor = ddpg_agent.actor_network
        self.critics = []
        self.state_inputs = []
        self.action_inputs = []
        self.t_state_inputs = []
        self.t_action_inputs = []

        self.verbose = verbose

        #TODO Build the rest of the subcritic system.
        # actor.layers = [state, output_layer1, output_layer2, output_layer3, ..., action = output_layer[-1]]
        for l, layer in enumerate(self.actor.layers[:-1]):
            state_dim = layer.get_shape()[1] # Only works for 1d
            action_dim = self.actor.layers[l+1].get_shape()[1]

            # Make a true and target placeholders (This is an optimization)
            state_input = tf.placeholder("float",[None,state_dim], name="subcritic_state_{}".format(l)) 
            action_input = tf.placeholder("float",[None,action_dim], name="subcritic_action_{}".format(l))
            t_state_input = tf.placeholder("float",[None,state_dim], name="subcritic_t_state_{}".format(l)) 
            t_action_input = tf.placeholder("float",[None,action_dim], name="subcritic_t_action_{}".format(l))

            self.state_inputs += [state_input]
            self.action_inputs += [action_input]
            self.t_state_inputs += [t_state_input]
            self.t_action_inputs += [t_action_input]
            b_size = tf.shape(state_input)[0]
            #Create critics for each input
            for j in range(action_dim):
                with tf.variable_scope("subcritic_l{}_n{}".format(l,j)):
                    self.critics.append(
                        PolynomialCritic(
                            self.sess,
                            state_input, tf.reshape(action_input[:,j], tf.pack([b_size, 1])),
                            t_state_input, tf.reshape(t_action_input[:,j], tf.pack([b_size, 1])),
                            order))

            # Create optimizer on all critics
            with tf.variable_scope("subcritic_training"):
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
