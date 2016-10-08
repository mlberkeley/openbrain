
import tensorflow as tf
import numpy as np
import math


LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01

class PolynomialCritic:
    def __init__(
            self,sess,
            state_placeholder,
            action_placeholder,
            t_state_placeholder,
            t_action_placeholder, order=2):
        """
        Creates a polynomial critic
        action-dim is always [1] considering that we are doing
        a critic per neuron
        """
        self.order = order
        self.sess = sess
        self.state_placeholder = state_placeholder
        self.action_placeholder = action_placeholder
        self.t_state_placeholder = t_state_placeholder
        self.t_action_placeholder = t_action_placeholder

        # create q network
        self.state_input,\
        self.action_input,\
        self.q_value_output,\
        self.net = self.create_poly_q(state_dim,action_dim)

        # create target q network (the same structure with q network)
        self.target_state_input,\
        self.target_action_input,\
        self.target_q_value_output,\
        self.target_update = self.create_target_q_network(state_dim,action_dim,self.net, order <= 1)

        self.create_loss()
        self.update_target()



    def create_loss(self):
        """ Define training optimizer """
        self.y_input = tf.placeholder("float",[None,1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

    def get_loss(self):
        return self.loss

    def create_poly_q(self):
        """ Initialize the polynomial critic network"""

        # Create the variables.
        linear = self.order <= 1
        layer_size = state_dim + action_dim if not linear else 1
        # TODO: ensure no conflict between dimension objects and ints
        W1 = self.variable([state_dim.value + action_dim, layer_size],state_dim.value)
        # might not want to hardcode the 1 if we want something like x^T (Wx + b)
        b1 = self.variable([1], state_dim.value)

        net = [W1, b1]
        q_value_output, net = self.setup_graph(self.state_placeholder , self.action_placeholder, net, linear)

        #   then let x =tf.concat(state_placeholder, action_placeholder) and the output of this polynomial
        #   critic will be Qn = x^TWx if order=2, or Qn = xW, if order =1, etc...
        return q_value_output, net


    def setup_graph(self, state_input, action_input, net, linear):
        """
        Sets up the network graph.
        """
        concat_input = tf.concat(1,[state_input, action_input])
        # TODO generalize this for order n (might be hard)
        if linear:
            q_value_output = tf.identity(tf.matmul(concat_input, net[0]) + net[1])
        else:
            # enforce symmetry of W
            W1 = 0.5 * (net[0] + tf.transpose(net[0]))
            xT = tf.transpose(concat_input)
            q_value_output = tf.identity(tf.matmul(xT, tf.matmul(W1,concat_input)) + net[1])
            net = [W1, net[1]]
        print("state_input {0}. action_input {1}".format(state_input, action_input))
        return q_value_output, net


    def create_target_q_network(self, net, linear):
        """ Initialize the target polynomial critic network"""
        # Implement
        state_input = tf.placeholder("float",[None,state_dim])
        action_input = tf.placeholder("float",[None,action_dim])


        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        q_value_output = self.setup_graph(state_input, action_input, target_net, linear)
        #   then let x =tf.concat(state_placeholder, action_placeholder) and
        return state_input, action_input, q_value_output, target_update

    def update_target(self):
        self.sess.run(self.target_update)

    def target_q(self,state_batch,action_batch):
        """
        Feeds the state and action batch to calculate
        the q value through the target network.
        """
        return self.sess.run(self.target_q_value_output,feed_dict={
            self.target_state_input:state_batch,
            self.target_action_input:action_batch
            })

    def q_value(self,state_batch,action_batch):
        """
        Feeds the state and action batch to calculate
        the q value through the regular network.
        """
        return self.sess.run(self.q_value_output,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch})
