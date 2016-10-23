import tensorflow as tf
from .layer import Layer

from .common.ou_noise import OUNoise

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

		self.layers = []
		self.layers += [Layer(self.sess, self.rewardInput, self.doneInput, \
							LAYER1_SIZE, state = self.stateInput, \
							nextState = self.nextStateInput, \
							stateDim = stateDim)]
		self.layers += [Layer(self.sess, self.rewardInput, self.doneInput, \
							LAYER2_SIZE, self.layers[0])]
		self.layers += [Layer(self.sess, self.rewardInput, self.doneInput, \
							actionDim, self.layers[1], activation=False)]

		self.actorOptimizer = self.createActorTraining()

		self.sess.run(tf.initialize_all_variables())
		self.explorationNoise = OUNoise(actionDim)

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
				 self.nextStateInput: nextStateBatch }
		return ops, feeds

	def getAction(self, stateInput):
		action = self.sess.run(self.layers[-1].output, feed_dict={
					self.stateInput: stateInput})
		action += self.explorationNoise.noise()
		return action[0]
