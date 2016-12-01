################################################################
# EXPERIMENT 2:
#
# In this experiment we will train an actor according to its subcritics
################################################################
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display
from itertools import chain
import gc
import argparse
gc.enable()

from brain.common.filter_env import makeFilteredEnv
from brain.brain import Brain

EXPLORE_TIME = -1

def test(env, agent, num_tests):
	"""
	Tests the agent.
	"""
	total_reward = 0
	for i in range(num_tests):
		state = env.reset()

		for j in range(env.spec.timestep_limit):
			#env.render()
			action = agent.getAction([state])
			state,reward,done,_ = env.step(action)
			total_reward += reward
			if done:
				break
	avg_reward = total_reward/num_tests
	return avg_reward


def run_experiment(exp_name, ENV_NAME='MountainCarContinuous-v0', EPISODES=10000, TEST=10):
	"""
	Runs the experiment.
	"""
	#env = makeFilteredEnv(gym.make(ENV_NAME))
	env = StupidGame()
	stateDim = 1#env.observation_space.shape[0]
	actionDim = 1#env.action_space.shape[0]
	sess = tf.Session()
	brain = Brain(sess, stateDim, actionDim)

	# Set up tensorboard.
	merged = tf.merge_all_summaries()
	train_writer = tf.train.SummaryWriter('/tmp/tboard/{}'.format(exp_name),
									  sess.graph)
	# To see graph run tensorboard --logdir=/tmp/exp1/tboard
	sess.run(tf.initialize_all_variables())

	t = 0
	for episode in range(EPISODES):
		state = env.reset()
		activations = None
		print("Episode: ", episode, end="")
		r_tot = 0

		for step in range(200):#env.spec.timestep_limit):
			t+= 1
			actions = brain.getAction(state)
			action = actions[-1][0]
			# Deal with the environment
			next_state,reward,done,_ = env.step(action)
			nextActions = brain.getAction(next_state)
			r_tot += reward

			ops, feeds = brain.perceive(reward, int(done), state, next_state, actions, nextActions, t > -1)
			# if episode %100 == 0:
			# 	env.render()
			if t > EXPLORE_TIME:
				if ops != None and feeds != None:
					ops = [merged] + ops
					result = sess.run(ops, feeds)
					#print(result[2])
					train_writer.add_summary(result[0], t)
			if done:
				break
			# Move on to next frame.
			state = next_state
		print(" ",  state, r_tot)

		# Testing:
		# if episode % 100 == 0 and episode > 100:
		# 	avg_reward = test(env, brain, TEST)
		# 	print(('episode: ',episode,'Evaluation Average Reward:',avg_reward))

class StupidGame:

	def __init__(self):
		self.reset()

	def reset(self):
		self.pose = 1
		return self.pose
	def step(self, action):
		self.pose += action
		#print(self.pose)
		if self.pose > 100:
			return self.pose, 100, True, None
		elif self.pose > 0:
			return self.pose, 1, False, None
		elif self.pose < -100:
			return self.pose, -100, True, None
		else:
			return self.pose, -1, False, None

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
	  '--name',
	  type=str,
	  default='tboard',
	  help="""\
	  the name of the experiment to run.\
	  """
	)
	args = parser.parse_args()
	run_experiment(args.name)
