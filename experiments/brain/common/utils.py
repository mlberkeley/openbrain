import tensorflow as tf
import math

TRACK_VARS=False

def variable_summaries(var, name):
	"""Attach a lot of summaries to a Tensor."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.scalar_summary('mean/' + name, mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.scalar_summary('stddev/' + name, stddev)
		tf.scalar_summary('max/' + name, tf.reduce_max(var))
		tf.scalar_summary('min/' + name, tf.reduce_min(var))
		tf.histogram_summary(name, var)

def variable(shape,f, name="Variable"):
    """
    Creates a tensor of SHAPE drawn from
    a random uniform distribution in 0 +/- 1/sqrt(f)
    """
    #TODO: fix this. currently shape is a [Dimension, int] object
    #v =  tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)), name=name)
    v = tf.Variable(tf.constant(-0.00001, shape=shape, name=name))
    if TRACK_VARS: variable_summaries(var, name)
    return v
