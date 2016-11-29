
import tensorflow as tf
from .layer import Layer

from .common.ou_noise import OUNoise
from .common.replay_buffer import ReplayBuffer

import numpy as np

LAYER1_SIZE = 1
LAYER2_SIZE = 1
ACTOR_LEARNING_RATE = 1e-4
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 32
BATCH_SIZE = 32
NUM_LAYERS = 1



class Brain:

	def __init__(self, sess, stateDim, actionDim):
		self.timestep = 0
		self.sess = sess
		self.stateInput = tf.placeholder("float", [1, stateDim])
		self.nextStateInput = tf.placeholder("float", [1, stateDim])
		self.rewardInput = tf.placeholder("float", [1]) 
		self.doneInput = tf.placeholder("float", [1])
		self.noises = [tf.placeholder("float", [1, size]) for size in [LAYER1_SIZE, actionDim]] #LAYER2_SIZE, actionDim]]
		self.actions = [tf.placeholder("float", [1, size])  for size in [LAYER1_SIZE, actionDim]]
		self.nextActions = [tf.placeholder("float", [1, size])  for size in [LAYER1_SIZE, actionDim]]

		self.stateDim = stateDim
		self.actionDim = actionDim

		self.layers = []
		self.layers += [Layer(self.sess, self.rewardInput, self.doneInput, \
							self.noises[0], LAYER1_SIZE, self.actions[0],  \
							self.nextActions[0],
							state = self.stateInput, \
							nextState = self.nextStateInput, \
							stateDim = stateDim)]
		#self.layers += [Layer(self.sess, self.rewardInput, self.doneInput, \
		#					self.noises[1], LAYER2_SIZE, self.layers[0])]
		#self.layers += [Layer(self.sess, self.rewardInput, self.doneInput, \
		#					self.noises[2], actionDim, self.layers[1], activation=False)]
		self.actorOptimizer = self.createActorTraining()

		self.sess.run(tf.initialize_all_variables())
		self.explorationNoises = [OUNoise(size) for size in [LAYER1_SIZE, actionDim]] # LAYER2_SIZE, actionDim]]
		self.actionDim = actionDim
		self.replayBuffer = ReplayBuffer(REPLAY_SIZE)

	def createActorTraining(self):
		grads_vars = []
		for layer in self.layers:
			grads_vars += [(-grad, var) for grad, var in zip(layer.grads, layer.weights)]
			#grads_vars += layer.l2grads_vars

		with tf.variable_scope('actor_learning'):
			optimizer = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE).apply_gradients(grads_vars)
		return optimizer

	def getCriticOps(self):
		ops = []
		for critic in self.layers:
			critic.updateTargets()
			ops += [critic.Qoptimizer]
		return ops

	def getTrain(self, reward, done, state, nextState, actions, nextActions, train_actor=True):
		ops = self.getCriticOps()
		if train_actor:
			ops += [self.actorOptimizer]
		feeds = {self.rewardInput: reward,
				 self.doneInput: done,
				 self.stateInput: state,
				 self.nextStateInput: nextState}
		for i, size in enumerate([LAYER1_SIZE, self.actionDim]): # LAYER2_SIZE, self.actionDim]):
			feeds[self.noises[i]] = [np.zeros(size)]
		feeds[self.actions[0]] = [actions[0]]
		feeds[self.nextActions[0]] = [nextActions[0]]
		return ops, feeds

	def getAction(self, stateInput):
		stateInput = np.reshape(stateInput, (1,1))
		feed_dict = {self.stateInput: stateInput}
		for i, noise in enumerate(self.explorationNoises):
			feed_dict[self.noises[i]] = [noise.noise()]
		actions = self.sess.run([self.layers[i].output for i in range(len(self.layers))], feed_dict=feed_dict)
		return actions

	def perceive(self, reward, done, state, nextState, actions, nextActions, train_actor=True):

		# self.replayBuffer.add(state, None, reward, nextState, done)

		# if self.replayBuffer.count() > REPLAY_START_SIZE:
		# 	minibatch = self.replayBuffer.get_batch(BATCH_SIZE)
		# 	stateBatch = np.asarray([data[0] for data in minibatch])
		# 	rewardBatch = np.asarray([data[2] for data in minibatch])
		# 	nextStateBatch = np.asarray([data[3] for data in minibatch])
		# 	doneBatch = np.asarray([data[4] for data in minibatch])

		state = np.reshape(state, [1, self.stateDim])
		nextState = np.reshape(nextState, [1, self.stateDim])
		reward = np.reshape(reward, [1])
		done = np.reshape(nextState, [1])
		actions = np.reshape(actions, [len(self.layers), self.actionDim])
		nextActions = np.reshape(nextActions, [len(self.layers), self.actionDim])
		return self.getTrain(reward, done, state, nextState, actions, nextActions, train_actor)
			
		if done:
			for noise in self.explorationNoises:
				noise.reset()
		return None, None