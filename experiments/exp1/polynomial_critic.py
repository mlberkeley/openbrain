
import tensorflow as tf
import numpy as np
import math


LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01

class PolynomialCritic:
	def __init__(self,sess,state_dim,action_dim):
		"""
			Creates a polynomial critic
			action-dim is always [1] considering that we are doing
			a critic per neuron
		"""
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		# create q network
		self.state_input,\
		self.action_input,\
		self.q_value_output,\
		self.net = self.create_poly_q(state_dim,action_dim)

		# create target q network (the same structure with q network)
		self.target_state_input,\
		self.target_action_input,\
		self.target_q_value_output,\
		self.target_update = self.create_target_q_network(state_dim,action_dim,self.net)

		self.create_training_method()

		# initialization
		self.sess.run(tf.initialize_all_variables())

		self.update_target()


	def create_training_method(self):
		# Define training optimizer
		self.y_input = tf.placeholder("float",[None,1])
		weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
		self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
		self.action_gradients = tf.gradients(self.q_value_output,self.action_input)

	def create_poly_q(self,state_dim,action_dim):
		state_input = tf.placeholder("float",[None,state_dim]) #The none is for batches!
		action_input = tf.placeholder("float",[None,action_dim])

		x = tf.concat(state_input, action_input) # Verify I am concating on the right dimensions
		# Here is an example for order 1
		W1 = self.variable([state_dim+action_dim,1 ],state_dim)
		b1 = self.variable([layer1_size],state_dim)
		
		q_value_output = tf.identity(tf.matmul(x1,W1) + b1)

		return state_input,action_input,q_value_output,[W1,b1]

	def create_target_q_network(self,state_dim,action_dim,net):
		# Implement
		pass

	def update_target(self):
		# Implement
		pass

	def train(self,y_batch,state_batch,action_batch):
		self.sess.run(self.optimizer,feed_dict={
			self.y_input:y_batch,
			self.state_input:state_batch,
			self.action_input:action_batch
			})

	def gradients(self,state_batch,action_batch):
		return self.sess.run(self.action_gradients,feed_dict={
			self.state_input:state_batch,
			self.action_input:action_batch
			})[0]

	def target_q(self,state_batch,action_batch):
		return self.sess.run(self.target_q_value_output,feed_dict={
			self.target_state_input:state_batch,
			self.target_action_input:action_batch
			})

	def q_value(self,state_batch,action_batch):
		return self.sess.run(self.q_value_output,feed_dict={
			self.state_input:state_batch,
			self.action_input:action_batch})

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
