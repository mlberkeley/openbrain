import tensorflow as tf
import numpy as np
import math
from critic_network import CriticNetwork
from replay_buffer import SubCriticReplayBuffer as ReplayBuffer


# Hyper Parameters
REPLAY_BUFFER_SIZE = 1000000
LAYER1_SIZE = 400
LAYER2_SIZE = 300
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
		if has_subcritics:
			self.subcritics = []
			# self.subcritics_replay_buffers = ReplayBuffer(REPLAY_BUFFER_SIZE)

		# create actor network
		self.state_input,self.action_output,self.net = self.create_network(state_dim,action_dim)

		# create target actor network
		self.target_state_input,self.target_action_output,self.target_update,self.target_net = self.create_target_network(state_dim,action_dim,self.net)

		# define training rules
		self.create_training_method()

		self.sess.run(tf.initialize_all_variables())

		self.update_target()

		#self.load_network()

	def create_training_method(self):
		self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
		self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net))

	def create_network(self,state_dim,action_dim):
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE

		state_input = tf.placeholder("float",[None,state_dim])

		W1 = self.variable([state_dim,layer1_size],state_dim)
		b1 = self.variable([layer1_size],state_dim)
		W2 = self.variable([layer1_size,layer2_size],layer1_size)
		b2 = self.variable([layer2_size],layer1_size)
		W3 = tf.Variable(tf.random_uniform([layer2_size,action_dim],-3e-3,3e-3))
		b3 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))

		layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)

		self.create_subcritic_network(state_dim, layer1_size, state_input, layer1)

		layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
		print("layer2 shape", layer2.get_shape(), layer1.get_shape())
		self.create_subcritic_network(layer1_size, layer2_size, layer1, layer2)

		action_output = tf.tanh(tf.matmul(layer2,W3) + b3)
		self.create_subcritic_network(layer2_size, int(action_output.get_shape()[1]), layer2, action_output)

		return state_input,action_output,[W1,b1,W2,b2,W3,b3]

	def create_subcritic_network(self, in_dim, out_dim, input_tensor, output_tensor):
		"""
		Create a subcritic network for the layer
		"""
		if self.has_subcritics:
			sc_replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
			self.subcritics.append([CriticNetwork(self.sess, in_dim, out_dim), output_tensor, sc_replay_buffer, (), input_tensor])


	#### Accessor methods for the subcritic network ####
	def get_sc_replay_buffer(self, subcritic_net):
		return subcritic_net[2]

	def get_sc_network(self, subcritic_net):
		return subcritic_net[0]

	def get_sc_output_tensor(self, subcritic_net):
		return subcritic_net[1]
	def get_sc_state_action(self, subcritic_net):
		return subcritic_net[3]
	def get_sc_input_tensor(self, subcritic_net):
		return subcritic_net[4]

	def update_sc_state_action(self, subcritic_net, new_state, new_action):
		""" Updates the last state/action for the sc network"""
		subcritic_net[3] = (new_state, new_action)

	def subcritics_perceive(self, env_next_state, reward, done):
		"""
		Add another perception to the subcritic network
		"""
		if self.has_subcritics:
			for sc in self.subcritics:
				state, action = self.get_sc_state_action(sc)
				self.action(env_next_state)
				next_state, next_action = self.get_sc_state_action(sc)
				replay_buffer = self.get_sc_replay_buffer(sc)
				replay_buffer.add(state,action,reward,next_state, next_action, done)


	def create_target_network(self,state_dim,action_dim,net):
		state_input = tf.placeholder("float",[None,state_dim])
		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]
		layer1 = tf.nn.relu(tf.matmul(state_input,target_net[0]) + target_net[1])
		layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + target_net[3])
		action_output = tf.tanh(tf.matmul(layer2,target_net[4]) + target_net[5])

		return state_input,action_output,target_update,target_net

	def update_target(self):
		self.sess.run(self.target_update)
		if self.has_subcritics:
			for sc in self.subcritics:
				net = self.get_sc_network(sc)
				net.update_target()

	def train(self,q_gradient_batch,state_batch):

		self.sess.run(self.optimizer,feed_dict={
			self.q_gradient_input:q_gradient_batch,
			self.state_input:state_batch
			})

		# TODO add training for the subnets
		# for sc in self.subcritics

	def actions(self,state_batch, reward_batch, done_batch):
		"""
		"""
		action_batch = self.sess.run(self.action_output,feed_dict={
			self.state_input:state_batch
			})

		if self.has_subcritics:
			for sc in self.subcritics:
				# Sample a random minibatch of N transitions from replay buffer
				replay_buffer = self.get_sc_replay_buffer(sc)
				minibatch = replay_buffer.get_batch(BATCH_SIZE)
				state_batch = np.asarray([data[0] for data in minibatch])
				action_batch = np.asarray([data[1] for data in minibatch])
				next_state_batch = np.asarray([data[3] for data in minibatch])
				next_action_batch = np.asarray([data[4] for data in minibatch])
				# for action_dim = 1

				net = self.get_sc_network(sc)
				action_batch = np.resize(action_batch,[BATCH_SIZE,net.action_dim])
				next_action_batch = np.resize(next_action_batch, [BATCH_SIZE, net.action_dim])
				print(next_action_batch)

				# TODO next_action_batch should actually be from the actor _target _network
				# if we were true to the proper paradigm
				q_value_batch = net.target_q(next_state_batch, next_action_batch)

				y_batch = []
				for i in range(BATCH_SIZE):
					if done_batch[i]:
						y_batch.append(reward_batch[i])
					else:
						y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
					y_batch = np.resize(y_batch, [BATCH_SIZE, 1])
				net.train(y_batch, state_batch, action_batch)
		return action_batch

	def action(self,state):
		""" Performs an action by propogating through the net"""
		action_output = self.sess.run(self.action_output,feed_dict={
			self.state_input:[state]
			})[0]
		self.save_sc_state_action()
		return action_output
	def save_sc_state_action(self):
		""" Save the state and action of the subcritic network """
		if self.has_subcritics:
			for sc in self.subcritics:
				## TODO this is still magic
				state = self.get_sc_input_tensor(sc)
				action = self.get_sc_output_tensor(sc)
				self.update_sc_state_action(sc, state, action)
	def update_sc_replay_buffers(self):
		#TODO finish this
		if self.has_subcritics:
			for sc in self.subcritics:
				# grab the input of the layer
				# grab the output of the layer
				pass




	def target_actions(self,state_batch):
		""" Lag actor network """

		next_action_batch = self.sess.run(self.target_action_output,feed_dict={
			self.target_state_input:state_batch
			})
		# TODO get the next_action for each subcritic on the network .
		# state will be the next_state. action will be the next_action
		if self.has_subcritics:
			pass
		#	 # TODO fix up for proper state_batch and action_batch
		#	 for sc in self.subcritics:
		#		 #Get the action batch and state batch
		#		 self.critic_network.target_q(next_state_batch,next_action_batch)
		return next_action_batch

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
'''
	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
			print "Could not find old network weights"
	def save_network(self,time_step):
		print 'save actor-network...',time_step
		self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

'''
