import tensorflow as tf
from .layer import Layer

from .common.ou_noise import OUNoise
import numpy as np

LAYER1_SIZE = 10
LAYER2_SIZE = 10
LEARNING_RATE = 1e-4

class Brain:

	def __init__(self, sess, stateDim, actionDim):
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

	def createActorTraining(self):
		grads_vars = []
		for layer in self.layers:
			grads_vars += [(-grad, var) for grad, var in zip(layer.grads, layer.weights)]

		with tf.variable_scope('actor_learning'):
			optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads_vars)
		return optimizer

	def getCriticOps(self):
		ops = []
		for critic in self.layers:
			critic.updateTargetCritic()
			ops += [critic.Qoptimizer]
		return ops

	def getTrain(self, rewardBatch, doneBatch, stateBatch, nextStateBatch):


		ops = self.getCriticOps()
		ops += [self.actorOptimizer]
		feeds = {self.rewardInput: rewardBatch,
				 self.doneInput: doneBatch,
				 self.stateInput: stateBatch,
				 self.nextStateInput: nextStateBatch}
		for i, size in enumerate([LAYER1_SIZE, LAYER2_SIZE, self.actionDim]):
			feeds[self.noises[i]] = [np.zeros(size)]
		return ops, feeds

	def getAction(self, stateInput):
		feed_dict = {self.stateInput: stateInput}
		for i, noise in enumerate(self.explorationNoises):
			feed_dict[self.noises[i]] = [noise.noise()]
		action = self.sess.run(self.layers[-1].output, feed_dict=feed_dict)
		return action[0]
