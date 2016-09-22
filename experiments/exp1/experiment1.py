################################################################
# EXPERIMENT 1:
#
# In this experiment we will train an actor according to a critic
# and then simultaneously learn and plot the resulting 
################################################################
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display
from itertools import chain
import gc
gc.enable()


import filter_env
import reward_env
import multi_ddpg

########################################
# All of this warrants a class :)
def init_data(cur_data, n_sub_critics):
	"""
	This is total trash. You can fix if you want.
	"""
	cur_data["episode"] = 0
	cur_data["rewards"] = [[]] # episode -> rewards
	cur_data["sub_critics"] = [[[]] for n in range(n_sub_critics)] # episode -> critics -> q_predicted
	cur_data["critic"] = [[]]

def new_episode_data(cur_data, n_sub_critics):
	"""
	This is total trash. You can fix if you want.
	"""
	cur_data["episode"] += 1
	cur_data["rewards"] += [[]]
	cur_data["sub_critics"] +=  [[[]] for cn in range(n_sub_critics.count)]
	cur_data["critic"] += [[]]

def record_data(cur_data, state, action, activations, reward, done, 
				agent, sub_critics):
	if not cur_data:
		init_data(cur_data, sub_critics.count)

	episode = cur_data["episode"]
	cur_data["rewards"][episode].append(reward)
	cur_data["sub_critics"][episode].append(sub_critics.q(activations))
	cur_data["critic"][episode].append(agent.critic_network.target_q([state],[action]))


	if done:
		new_episode_data(cur_dat)
################

def test(env, agent, test):
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
	return test


def run_experiment(ENV_NAME='MountainCarContinuous-v0', EPISODES=10000, TEST=10):
	env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
	cur_data = {}

	# Create the standard DDPG agent.
	agent = DDPG(env,)
	sub_critics = SubCritics(agent, order=1) # Make linear (order 1) subcritics

	for episode in range(EPISODES):
		state = env.reset()
		print("Episode: ", episode, endn="")
		r_tot = 0

		for step in range(env.spec.timestep_limit):
			# Explore state space.
			action, activations = agent.noise_action_activations(state)

			# Deal with the environment
			next_state,reward,done,_ = env.step(action)
			r_tot += reward
			env.render()

			# Train subcrticis
			subcritics.perceive(activations, reward, done)

			# Train DDPG
			agent.perceive(state,action,reward,next_state,done)

			record_data(
				cur_data, state, action, activations, reward, done, 
				agent, subcritics)
			# plot_data(cur_data)

			if done:
				break
			# Move on to next frame.
			state = next_state
		print(rtot)

		# Testing:
		if episode % 100 == 0 and episode > 100:
			test(env, agent, TEST)
			print(('episode: ',episode,'Evaluation Average Reward:',ave_reward))
