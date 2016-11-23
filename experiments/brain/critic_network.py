
import tensorflow as tf
import numpy as np
import math

from .common.utils import variable
from .common.utils import variable_summaries

LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01
GAMMA = 0.99

class CriticNetwork:
	"""docstring for CriticNetwork"""
	def __init__(self, state_dim, action_dim, state, action, reward, next_state, next_action, done):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.state_input = state
		self.action_input = action
		self.reward_input = reward
		self.next_state_input =next_state
		self.next_action_input = next_action
		self.done_input = done

		# create q network\
		with tf.variable_scope("Q"):
			self.q_value_output,\
			self.net = self.create_q_network(state_dim, action_dim)

		# create target q network (the same structure with q network)
		with tf.variable_scope("Target"):
			self.target_q_value_output,\
			self.target_update = self.create_target_q_network(state_dim, action_dim, self.net)

		self.create_training_method()


	def create_training_method(self):
		# Define training optimizer
		weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
		diff = self.q_value_output - self.reward_input - tf.matmul(
			tf.diag(1- self.done_input), tf.scalar_mul(GAMMA, self.target_q_value_output))
		loss = tf.square(diff)
		loss = tf.Print(loss, [loss, diff, (1- self.done_input), tf.diag(1- self.done_input), self.reward_input,
			self.q_value_output])
		self.cost = tf.reduce_mean(loss) + weight_decay
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

	def create_q_network(self,state_dim,action_dim):
		# the layer size could be changed
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE

		state_input = self.state_input
		action_input = self.action_input

		W1 = variable([state_dim,layer1_size],self.state_dim)
		b1 = variable([layer1_size],self.state_dim)
		W2 = variable([layer1_size,layer2_size],layer1_size+self.action_dim)
		W2_action = variable([self.action_dim,layer2_size],layer1_size+self.action_dim)
		b2 = variable([layer2_size],layer1_size+self.action_dim)
		W3 = tf.Variable(tf.random_uniform([layer2_size,1],-3e-3,3e-3))
		b3 = tf.Variable(tf.random_uniform([1],-3e-3,3e-3))

		layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
		layer2 = tf.nn.relu(tf.matmul(layer1,W2) + tf.matmul(action_input,W2_action) + b2)
		q_value_output = tf.identity(tf.matmul(layer2,W3) + b3)
		variable_summaries(q_value_output, "Critic_Q")
		return q_value_output,[W1,b1,W2,W2_action,b2,W3,b3]

	def create_target_q_network(self,state_dim,action_dim,net):
		state_input = self.next_state_input
		action_input = self.next_action_input

		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		layer1 = tf.nn.relu(tf.matmul(state_input,target_net[0]) + target_net[1])
		layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + tf.matmul(action_input,target_net[3]) + target_net[4])
		q_value_output = tf.identity(tf.matmul(layer2,target_net[5]) + target_net[6])
		variable_summaries(q_value_output, "Critic_T_Q")
		return q_value_output,target_update