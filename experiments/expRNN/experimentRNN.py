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


import common.filter_env
from ddpg import DDPG
from sub_critics import SubCritics


# IN THIS IMPLEMENTATION, it takes NETWORK_TIME iterations of the network to make one action of the state
N_NEURONS = 1000
NETWORK_TIME = 5

########################################
# All of this warrants a class :)

# def init_data(cur_data, n_sub_critics):
# 	"""
# 	This is total trash. You can fix if you want.
# 	"""
# 	cur_data["episode"] = 0
# 	cur_data["rewards"] = [[]] # episode -> rewards
# 	cur_data["sub_critics"] = [[[]] for n in range(n_sub_critics)] # episode -> critics -> q_predicted
# 	cur_data["critic"] = [[]]
#
# def new_episode_data(cur_data, n_sub_critics):
# 	"""
# 	This is total trash. You can fix if you want.
# 	"""
# 	cur_data["episode"] += 1
# 	cur_data["rewards"] += [[]]
# 	cur_data["sub_critics"] +=  [[[]] for cn in range(n_sub_critics)]
# 	cur_data["critic"] += [[]]
#
# def record_data(cur_data, state, action, activations, reward, done,
# 				agent, sub_critics):
# 	if not cur_data:
# 		init_data(cur_data, sub_critics.count)
#
# 	episode = cur_data["episode"]
# 	cur_data["rewards"][episode].append(reward)
# 	cur_data["sub_critics"][episode].append(sub_critics.q(activations))
# 	cur_data["critic"][episode].append(agent.critic_network.target_q([state],[action]))
#
#
# 	if done:
# 		new_episode_data(cur_data, sub_critics.count)


################


def test(env, agent, TEST):
	"""
	Tests the agent.
	"""
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


def run_experiment(ENV_NAME='MountainCarContinuous-v0', EPISODES=10000, TEST=10, NETWORK_TIME=NETWORK_TIME):
	"""
	Runs the experiment on the target en
	"""
	env = common.filter_env.makeFilteredEnv(gym.make(ENV_NAME))
	cur_data = {}

	# actual state, action dimensions of space
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	assert N_NEURONS > state_dim
	assert N_NEURONS > action_dim

	# Create the standard DDPG agent.
	agent = DDPG(env, N_NEURONS)
	sub_critics = SubCritics(agent, order=1) # Make linear (order 1) subcritics

	for episode in range(EPISODES):
		state = env.reset()

		# initial voltages: zero
		state = np.append(state, np.zeros(N_NEURONS - state_dim))

		print("Episode: ", episode, end="")
		r_tot = 0

		for step in range(env.spec.timestep_limit):

			for t in range(NETWORK_TIME):

				action, activations = agent.noise_action_activations(state)
				print("activations: ", activations)

				if t == 0:
					a = action[-action_dim:]
					print("Action: ", a)
					n_s,reward,done,_ = env.step(a)
					r_tot += reward
					print("Env. State: ", n_s)
					env.render()


				# next state: voltages!
				next_state = np.array(action) + np.append(n_s, np.zeros(N_NEURONS - state_dim))

				# Train subcrticis
				sub_critics.perceive(activations, reward/NETWORK_TIME, done)

				# Train DDPG
				agent.perceive(state, action, reward/NETWORK_TIME, next_state, done)

				#record_data(
				#	cur_data, state, action, activations, reward, done,
				#	agent, sub_critics)
				# plot_data(cur_data)

				if done:
					break


				# Move on to next frame.
				state = next_state


		print(" ", r_tot)

		# Testing:
		if episode % 100 == 0 and episode > 100:
			test(env, agent, TEST)
			print(('episode: ',episode,'Evaluation Average Reward:',ave_reward))

if __name__ == '__main__':
    run_experiment()