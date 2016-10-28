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



class Brain:

	def __init__(self, sess, stateDim, actionDim):
		self.timestep = 0
		self.sess = sess
		self.stateInput = tf.placeholder("float", [None, stateDim])
		self.nextStateInput = tf.placeholder("float", [None, stateDim])
		self.rewardInput = tf.placeholder("float", [None]) 
		self.doneInput = tf.placeholder("float", [None])
		self.noises = [tf.placeholder("float", [None, size]) for size in [LAYER1_SIZE, LAYER2_SIZE, actionDim]]


		self.layers = []
		self.layers += [Layer(self.sess, self.rewardInput, self.doneInput, \
							self.noises[0], LAYER1_SIZE, state = self.stateInput, \
							nextState = self.nextStateInput, \
							stateDim = stateDim)]
		self.layers += [Layer(self.sess, self.rewardInput, self.doneInput, \
							self.noises[1], LAYER2_SIZE, self.layers[0])]
		self.layers += [Layer(self.sess, self.rewardInput, self.doneInput, \
							self.noises[2], actionDim, self.layers[1], activation=False)]

		self.actorOptimizer = self.createActorTraining()

		self.sess.run(tf.initialize_all_variables())
		self.explorationNoises = [OUNoise(size) for size in [LAYER1_SIZE, LAYER2_SIZE, actionDim]]
		self.actionDim = actionDim
		self.replayBuffer = ReplayBuffer(REPLAY_SIZE)

	def createActorTraining(self):
		grads_vars = []
		for layer in self.layers:
			grads_vars += [(-grad, var) for grad, var in zip(layer.grads, layer.weights)]
			grads_vars += layer.l2grads_vars

		with tf.variable_scope('actor_learning'):
			optimizer = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE).apply_gradients(grads_vars)
		return optimizer

	def getCriticOps(self):
		ops = []
		for critic in self.layers:
			critic.updateTargetCritic()
			ops += [critic.Qoptimizer]
		return ops

	def getTrain(self, rewardBatch, doneBatch, stateBatch, nextStateBatch, train_actor=True):
		ops = self.getCriticOps()
		if train_actor:
			ops += [self.actorOptimizer]
		feeds = {self.rewardInput: rewardBatch,
				 self.doneInput: doneBatch,
				 self.stateInput: stateBatch,
				 self.nextStateInput: nextStateBatch}
		for i, size in enumerate([LAYER1_SIZE, LAYER2_SIZE, self.actionDim]):
			feeds[self.noises[i]] = [np.zeros(size)]
		return ops, feeds

	def getAction(self, stateInput):
		stateInput = np.reshape(stateInput, (1,1))
		feed_dict = {self.stateInput: stateInput}
		for i, noise in enumerate(self.explorationNoises):
			feed_dict[self.noises[i]] = [noise.noise()]
		action = self.sess.run(self.layers[-1].output, feed_dict=feed_dict)
		return action[0]

	def perceive(self, reward, done, state, nextState, train_actor=True):

		self.replayBuffer.add(state, None, reward, nextState, done)

		if self.replayBuffer.count() > REPLAY_START_SIZE:
			minibatch = self.replayBuffer.get_batch(BATCH_SIZE)
			stateBatch = np.asarray([data[0] for data in minibatch])
			rewardBatch = np.asarray([data[2] for data in minibatch])
			nextStateBatch = np.asarray([data[3] for data in minibatch])
			doneBatch = np.asarray([data[4] for data in minibatch])

			stateBatch = np.reshape(stateBatch, [BATCH_SIZE, 1])
			nextStateBatch = np.reshape(nextStateBatch, [BATCH_SIZE, 1])
			rewardBatch = np.reshape(stateBatch, [BATCH_SIZE])
			doneBatch = np.reshape(nextStateBatch, [BATCH_SIZE])
			return self.getTrain(rewardBatch, doneBatch, stateBatch, nextStateBatch, train_actor)
			
		if done:
			for noise in self.explorationNoises:
				noise.reset()
		return None, None