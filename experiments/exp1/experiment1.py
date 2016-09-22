################################################################
# EXPERIMENT 1:
#
# In this experiment we will train an actor according to a critic
# and then simultaneously learn and plot the resulting 
################################################################

import matplotlib.pyplot as plt
from IPython import display
from itertools import chain
import importlib

import filter_env
import reward_env
import multi_ddpg
import gym
import numpy as np
importlib.reload(multi_ddpg)
import gc
import tensorflow as tf
gc.enable()



def main(ENV_NAME='MountainCarContinuous-v0', EPISODES=10000, TEST=10):
	env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))

	# Create the standard DDPG agent.
	agent = DDPG(env)

	for episode in range(EPISODES):
		state = env.reset()
		#print "episode:",episode
		# Train

		for step in range(env.spec.timestep_limit):
			action, voltage = agent.noise_action_voltage(state)
			next_state,reward,done,_ = env.step(action)
			env.render()
			agent.perceive(state,voltage,action,reward,next_state,done)
			state = next_state
			if done:
				break
		# Testing:
		if episode % 100 == 0 and episode > 100:
			total_reward = 0
			for i in range(TEST):
				state = env.reset()
				for j in range(env.spec.timestep_limit):
					#env.render()
					action = agent.action(state) # direct action for test
					state,reward,done,_ = env.step(action)
					total_reward += reward
					if done:
						break
			ave_reward = total_reward/TEST
			print(('episode: ',episode,'Evaluation Average Reward:',ave_reward))
	env.monitor.close()

if __name__ == '__main__':
	main()
