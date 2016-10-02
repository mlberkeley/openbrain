import tensorflow as tf
import numpy as np
import math

# Hyper Parameters
REPLAY_BUFFER_SIZE = 1000000
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64
GAMMA = 0.99

class ActorNetwork:
	"""docstring for ActorNetwork"""
	def __init__(self,sess,state_dim,action_dim,has_subcritics=False):
		#"""Constructor
		#param: subcritics : flag to turn on subcritic networks (critic networks per Q layer) """
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.has_subcritics = has_subcritics
		self.layers = []
		self.target_layers = []
		# create actor network
		self.state_input,self.action_output,self.net = self.create_network(state_dim,action_dim)

		# create target actor network
		self.target_state_input,\
		self.target_action_output,\
		self.target_update,\
		self.target_net = self.create_target_network(state_dim,action_dim,self.net)

		# define training rules
		self.create_training_method()

		self.sess.run(tf.initialize_all_variables())

		self.update_target()

		#self.load_network()

	def create_training_method(self):
		self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
		self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(list(zip(self.parameters_gradients,self.net)))

	def create_network(self,state_dim,action_dim):

		state_input = tf.placeholder("float",[None,state_dim])
		self.layers += [tf.identity(state_input)]

		W = tf.Variable(tf.random_uniform([state_dim,action_dim],-3e-3,3e-3))
		b = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))

		action_output = tf.tanh(tf.matmul(state_input,W) + b)
		self.layers += [action_output]

		return state_input,action_output,[W,b]


	def create_target_network(self,state_dim,action_dim,net):
		state_input = tf.placeholder("float",[None,state_dim])
		self.target_layers += [tf.identity(state_input)]
		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]
		action_output = tf.tanh(tf.matmul(state_input,target_net[0]) + target_net[1])
		self.target_layers += [action_output]

		return state_input,action_output,target_update,target_net

	def update_target(self):
		self.sess.run(self.target_update)

	def train(self,q_gradient_batch,state_batch):

		self.sess.run(self.optimizer,feed_dict={
			self.q_gradient_input:q_gradient_batch,
			self.state_input:state_batch
			})

	def actions(self,state_batch):
		"""
		"""
		action_batch = self.sess.run(self.action_output,feed_dict={
			self.state_input:state_batch
			})
		return action_batch

	def action(self,state):
		""" Performs an action by propogating through the net"""
		action_output = self.sess.run(self.action_output,feed_dict={
			self.state_input:[state]
			})[0]
		return action_output


	def action_activations(self, state):
		""" Gets a pair of action, [state, layer1, layer2, ...] """
		output = self.sess.run([self.action_output,
			self.layers],feed_dict={
			self.state_input:[state]
			})
		action = output[0][0]
		activations = [activation[0] for activation in output[1]]
		return action, activations


	def target_actions(self,state_batch):
		""" Lag actor network """

		next_action_batch = self.sess.run(self.target_action_output,feed_dict={
			self.target_state_input:state_batch
			})
		return next_action_batch

	def target_action_activations(self, state_batch):
		next_action_batch = self.sess.run(
			[self.target_action_output,self.target_layers],feed_dict={
			self.target_state_input:state_batch
			})

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
