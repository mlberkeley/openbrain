import tensorflow as tf

from .common.utils import variable, variable_summaries

LEARNING_RATE = 1e-2
TAU = 0.0001
GAMMA = 0.9
ALPHA = 0.3


class Layer:

	def __init__(self, sess, reward, done, noise, size, prevLayer=None, state=None, \
					nextState =None, stateDim=None, activation=True):
		self.sess = sess
		self.reward = reward
		self.done = done
		self.noise = noise
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
			self.output, self.targetOutput, self.weights = self.construct()
			variable_summaries(self.output, self.name + "/action")
			variable_summaries(self.reward, self.name + "/reward")
			with tf.variable_scope('subcritic'):
				self.Q, self.Qweights = self.createCritic()
				self.Qtarget, self.Qupdate = self.createTargetCritic()
				self.Qloss = self.createCriticLoss()
				self.Qoptimizer = self.createCriticLearning()
			self.grads = tf.gradients(self.Q, self.weights)
			self.l2grads_vars = self.computeL2RegGrads()
	def construct(self):
		with tf.variable_scope('actor'):
			W = variable([self.prevSize, self.size], self.prevSize, name='weights')
			b = variable([self.size], self.prevSize, name='bias')
			variable_summaries(b, self.name + "/bias")
			variable_summaries(W, self.name + "/weights")
			output = tf.matmul(self.input, W) + b + self.noise
			if self.activation:
				output = tf.nn.tanh(output)
		with tf.variable_scope('actor_target'):
			targetOutput = tf.matmul(self.targetInput, W) + b + self.noise
			if self.activation:
				targetOutput = tf.nn.tanh(targetOutput)
		return output, targetOutput, [W, b]

	def createCritic(self):
		with tf.variable_scope('Q'):
			Ms = variable([self.prevSize, self.size], self.prevSize)
			Ma = variable([self.size, 1], self.prevSize)
			b = variable([self.size], self.prevSize)

			q = tf.identity(tf.matmul(self.input, Ms) + tf.matmul(self.output, Ma) + b)
			variable_summaries(q, self.name + "/Q")
		return q, [Ms, Ma, b]

	def createTargetCritic(self):
		with tf.variable_scope('Q_target'):
			ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
			update = ema.apply(self.Qweights)
			weights = [ema.average(x) for x in self.Qweights]

			q = tf.identity(tf.matmul(self.targetInput, weights[0]) \
									 + tf.matmul(self.targetOutput, weights[1]) \
									 + weights[2])
			variable_summaries(q, self.name + "/Qtarget")
		return q, update

	def updateTargetCritic(self):
		self.sess.run(self.Qupdate)

	def createCriticLoss(self):
		with tf.variable_scope('loss'):
			reward = tf.transpose([self.reward for _ in range(self.size)])
			diff = self.Q - reward - tf.matmul(tf.diag(self.done), tf.scalar_mul(GAMMA, self.Qtarget))
			loss = tf.square(diff)
			# l2reg = ALPHA * tf.square(tf.concat(1, [tf.unpack(self.Qweights[0]), \
			# 										 tf.unpack(tf.transpose(self.Qweights[1]))]))
			# loss = tf.concat(0, [l, l2reg])
			variable_summaries(diff, self.name + "/diff")
			variable_summaries(loss, self.name + "/loss")
		return loss

	def createCriticLearning(self):
		with tf.variable_scope('learning'):
			gradLQ = tf.gradients(self.Qloss, self.Q)
			gradQM = tf.gradients(self.Q, self.Qweights, gradLQ)
			optimizer = tf.train.AdamOptimizer(LEARNING_RATE) \
						.apply_gradients(list(zip(gradQM, self.Qweights)))
		return optimizer

	def computeL2RegGrads(self):

		optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
		loss = tf.concat(0, [self.weights[0], [self.weights[1]]])
		print(loss)
		return optimizer.compute_gradients(loss, [self.weights[0], self.weights[1]])