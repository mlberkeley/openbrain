import tensorflow as tf
from .layer import Layer

from .common.ou_noise import OUNoise
from .common.replay_buffer import ReplayBuffer
from .critic_network import CriticNetwork

import numpy as np

LAYER1_SIZE = 2
LAYER2_SIZE = 4
ACTOR_LEARNING_RATE = 1e-2
SUBCRITIC_LEARNING_RATE = 2
REPLAY_SIZE = 1000000
REPLAY_START_SIZE = 30000
BATCH_SIZE = 128

class GlobalBrain:

	def __init__(self, sess, stateDim, actionDim):
		self.sess = sess
		self.stateDim = stateDim
		self.actionDim = actionDim

		# Set up place holders
		self.stateInput = tf.placeholder("float", [None, stateDim])
		self.nextStateInput = tf.placeholder("float", [None, stateDim])
		self.actionInput = tf.placeholder("float", [None, actionDim])
		self.rewardInput = tf.placeholder("float", [None])
		self.doneInput = tf.placeholder("float", [None])

		# Construct a series of layers
		self.layers = []
		self.layers += [Layer(self.sess, LAYER1_SIZE, state = self.stateInput, \
							nextState = self.nextStateInput, \
							stateDim = stateDim)]
		self.layers += [Layer(self.sess, LAYER2_SIZE, self.layers[0])]
		self.layers += [Layer(self.sess, actionDim, self.layers[1], activation=True)]

		self.next_action = self.layers[-1].targetOutput

		# Create the Global Critic.
		with tf.variable_scope("global_critic"):
			self.critic = CriticNetwork(
				stateDim,actionDim,
				self.stateInput, self.actionInput,
				self.rewardInput, self.nextStateInput,
				self.next_action,
				self.doneInput)

		# Create the optiomizers
		self.criticOptimizer = self.critic.optimizer
		self.subcriticOptimizer = self.createSubcriticTraining()
		self.actorOptimizer = self.createActorTraining()

		self.explorationNoise = OUNoise(actionDim)
		self.replayBuffer = ReplayBuffer(REPLAY_SIZE)

	def createActorTraining(self):
		grads_vars = []
		for layer in self.layers:
			grads_vars += layer.actor_grad_vars

		with tf.variable_scope('actor_learning'):
			optimizer = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE).apply_gradients(grads_vars)
		return optimizer

	def createSubcriticTraining(self):
		"""
		Force all layer Q values to match the global Q.
		"""
		grad_vars = []
		with tf.variable_scope('subcritic_learning'):
			for layer in self.layers:
				with tf.variable_scope(layer.name):
					grad_vars += layer.createCriticTraining(self.critic.q_TD)

			optimizer = tf.train.AdamOptimizer(SUBCRITIC_LEARNING_RATE).apply_gradients(grad_vars)
		return optimizer


	def getTrain(self, rewardBatch, doneBatch, stateBatch, nextStateBatch, actionBatch, train_actor=True):
		ops =  [self.critic.target_update]
		ops += [layer.target_update for layer in self.layers]
		ops+=  [self.criticOptimizer,
			self.subcriticOptimizer]

		if train_actor:
			ops += [self.actorOptimizer]
		feeds = {self.rewardInput: rewardBatch,
				 self.doneInput: doneBatch,
				 self.stateInput: stateBatch,
				 self.nextStateInput: nextStateBatch,
				 self.actionInput: actionBatch}
		return ops, feeds

	def getAction(self, stateInput):
		feed_dict = {self.stateInput: [stateInput]}
		action = self.sess.run(self.layers[-1].output, feed_dict=feed_dict)
		return action[0] + self.explorationNoise.noise()

	def perceive(self, reward, done, state, action, nextState, train_actor=True):

		self.replayBuffer.add(state, action, reward, nextState, done)

		if self.replayBuffer.count() > REPLAY_START_SIZE:
			minibatch = self.replayBuffer.get_batch(BATCH_SIZE)
			stateBatch = np.asarray([data[0] for data in minibatch])
			actionBatch = np.asarray([data[1] for data in minibatch])
			rewardBatch = np.asarray([data[2] for data in minibatch])
			nextStateBatch = np.asarray([data[3] for data in minibatch])
			doneBatch = np.asarray([data[4] for data in minibatch])
			return self.getTrain(rewardBatch, doneBatch, stateBatch, nextStateBatch, actionBatch, train_actor)

		if done:
			self.explorationNoise.reset()
		return None, None
