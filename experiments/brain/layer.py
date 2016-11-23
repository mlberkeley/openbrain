import tensorflow as tf

from .common.utils import variable, variable_summaries

TAU = 0.001
GAMMA = 0.9
ALPHA = 0.01

class Layer:

	def __init__(self, sess, size, prevLayer=None, state=None, \
					nextState =None, stateDim=None, activation=True):
		self.sess = sess
		self.size = size
		self.activation = activation
		if prevLayer:
			self.num = prevLayer.num + 1
			self.prevSize = prevLayer.size
			self.input = prevLayer.output
			self.targetInput = prevLayer.targetOutput
		else:
			self.num = 0
			self.prevSize = stateDim
			self.input = state
			self.targetInput = nextState
		self.name = "layer_{}".format(self.num)

		with tf.variable_scope(self.name):
			self.output, self.targetOutput, self.weights, self.target_update = self.construct()
			variable_summaries(self.output, self.name + "/action")
			with tf.variable_scope('subcritic'):
				self.Q, self.Qweights = self.createCritic()
			self.actor_grad_vars = list(zip(tf.gradients(-self.Q, self.weights), self.weights))

	def construct(self):
		"""
		Constructs the actor and the target actor for the layer.
		"""
		with tf.variable_scope('actor'):
			W = variable([self.prevSize, self.size], self.prevSize, name='weights')
			b = variable([self.size], self.prevSize, name='bias')
			variable_summaries(b, self.name + "/bias")
			variable_summaries(W, self.name + "/weights")

			output = tf.matmul(self.input, W) + b
			if self.activation:
				output = tf.nn.tanh(output)

		with tf.variable_scope('actor_target'):
			# TODO MOVE THE TRANSFER GLOBALLY.
			with tf.variable_scope("transfer"):
				ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
				target_update = ema.apply([W, b])
				Wt,bt = ema.average(W), ema.average(b)
			targetOutput = tf.matmul(self.targetInput, Wt) + bt
			print(targetOutput)
			if self.activation:
				targetOutput = tf.nn.tanh(targetOutput)
		return output, targetOutput, [W, b], target_update

	def createCritic(self):
		with tf.variable_scope('Q'):
			
			Ms = variable([self.prevSize, self.size], self.prevSize)
			Ma = variable([self.size, 1], self.prevSize)
			b = variable([self.size], self.prevSize)

			q = tf.identity(tf.matmul(self.input, Ms) + tf.matmul(self.output, Ma) + b)
			variable_summaries(q, self.name + "/Q")
		return q, [Ms, Ma, b]

	def createCriticTraining(self, true_Q):
		self.Qloss = self.createCriticLoss(true_Q)
		gradLQ = tf.gradients(self.Qloss, self.Q)
		gradQM = tf.gradients(self.Q, self.Qweights, gradLQ)
		return list(zip(gradQM, self.Qweights))

	def createCriticLoss(self, true_Q):
		with tf.variable_scope('loss'):
			l = tf.square(self.Q - true_Q)
			loss = tf.reduce_mean(l) + ALPHA*(
				tf.nn.l2_loss(self.Qweights[0]) 
				+ tf.nn.l2_loss(self.Qweights[1] )
				+ tf.nn.l2_loss(self.Qweights[2]))
			variable_summaries(loss, self.name + "/loss")
		return loss