
import tensorflow as tf
import numpy as np
import math


LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01

class PolynomialCritic:
	def __init__(self,sess,state_dim,action_dim=1, order=1):
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
        layer1_size = 1

        state_input = tf.placeholder("float",[None,state_dim]) #The none is for batches!
		action_input = tf.placeholder("float",[None,action_dim])

        # TODO Verify I am concating on the right dimensions
		concat_input = tf.concat(state_input, action_input)
		# Here is an example for order 1
        # TODO generalize this for order n
		W1 = self.variable([state_dim + action_dim, layer1_size],state_dim)
		b1 = self.variable([layer1_size], state_dim)

        # TODO generalize this for order n
		q_value_output = tf.identity(tf.matmul(concat_input, W1) + b1)
        #   then let x =tf.concat(state_placeholder, action_placeholder) and the output of this polynomial
		#   critic will be Qn = x^TWx if order=2, or Qn = xW, if order =1, etc...
		return state_input,action_input,q_value_output,[W1,b1]

	def create_target_q_network(self,state_dim,action_dim,net):
		# Implement
        state_input = tf.placeholder("float",[None,state_dim])
		action_input = tf.placeholder("float",[None,action_dim])
        # TODO Verify I am concating on the right dimensions
        concat_input = tf.concat(state_input, action_input)

        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_udpate = ema.apply(net)
        target_net = [ema.average(x) for x in net]
        # Here is an example for order 1
        # TODO generalize this for order n
        q_value_output = tf.identity(tf.matmul(concat_input, target_net[0]) + target_net[1])
        #   then let x =tf.concat(state_placeholder, action_placeholder) and the output of this polynomial
		#   critic will be Qn = x^TWx if order=2, or Qn = xW, if order =1, etc...
        return state_input, action_input, q_value_output, target_update

	def update_target(self):
		self.sess.run(self.target_update)

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
