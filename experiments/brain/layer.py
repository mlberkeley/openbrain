import tensorflow as tf

from .common.utils import variable, variable_summaries

LEARNING_RATE = 1e-3
TAU = 0.001
GAMMA = 0.9

class Layer:

	def __init__(self, sess, reward, done, size, prevLayer=None, state=None, \
					nextState =None, stateDim=None):
		self.sess = sess
		self.reward = reward
		self.done = done
		self.size = size
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
			variable_summaries(self.output, self.name + "_action")
			with tf.variable_scope('subcritic'):
				self.Q, self.Qweights = self.createCritic()
				self.Qtarget, self.Qupdate = self.createTargetCritic()
				self.Qloss = self.createCriticLoss()
				self.Qoptimizer = self.createCriticLearning()
			self.grads = tf.gradients(self.Q, self.weights)

		self.sess.run(tf.initialize_all_variables())

	def construct(self):
		with tf.variable_scope('actor'):
			W = variable([self.prevSize, self.size], self.prevSize)
			b = variable([self.size], self.prevSize)
			output = tf.nn.relu(tf.matmul(self.input, W) + b)
		with tf.variable_scope('actor_target'):
			targetOutput = tf.nn.relu(tf.matmul(self.targetInput, W) + b)
		return output, targetOutput, [W, b]

	def createCritic(self):
		with tf.variable_scope('Q'):
			Ms = variable([self.prevSize, self.size], self.prevSize)
			Ma = variable([self.size, 1], self.prevSize)
			b = variable([self.size], self.prevSize)

			q = tf.identity(tf.matmul(self.input, Ms) + tf.matmul(self.output, Ma) + b)
			variable_summaries(q, self.name + "_Q")
		return q, [Ms, Ma, b]

	def createTargetCritic(self):
		with tf.variable_scope('Q_target'):
			ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
			update = ema.apply(self.Qweights)
			weights = [ema.average(x) for x in self.Qweights]

			q = tf.identity(tf.matmul(self.targetInput, weights[0]) \
									 + tf.matmul(self.targetOutput, weights[1]) \
									 + weights[2])
			variable_summaries(q, self.name + "_Qtarget")
		return q, update

	def updateTargetCritic(self):
		self.sess.run(self.Qupdate)

	def createCriticLoss(self):
		with tf.variable_scope('loss'):
			loss = tf.square(self.Q \
							   - self.reward \
							   - tf.mul(self.done, tf.scalar_mul(GAMMA, self.Qtarget)))
			variable_summaries(loss, self.name + "_loss")
		return loss

	def createCriticLearning(self):
		with tf.variable_scope('learning'):
			gradLQ = tf.gradients(self.Qloss, self.Q)
			gradQM = tf.gradients(self.Q, self.Qweights, gradLQ)
			optimizer = tf.train.AdamOptimizer(LEARNING_RATE) \
						.apply_gradients(list(zip(gradQM, self.Qweights)))
		return optimizer