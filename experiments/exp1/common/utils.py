import tensorflow as tf

def variable(shape,f):
    """
    Creates a tensor of SHAPE drawn from
    a random uniform distribution in 0 +/- 1/sqrt(f)
    """
    #TODO: fix this. currently shape is a [Dimension, int] object
    return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
