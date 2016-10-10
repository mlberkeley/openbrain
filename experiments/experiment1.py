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

from brain import DDPG
from brain import SubCritics
from brain.common.filter_env import makeFilteredEnv


def test(env, agent, num_tests):
    """
    Tests the agent.
    """
    total_reward = 0
    for i in range(num_tests):
        state = env.reset()
        for j in range(env.spec.timestep_limit):
            #env.render()
            action = agent.action(state) # direct action for test
            state,reward,done,_ = env.step(action)
            total_reward += reward
            if done:
                break
    avg_reward = total_reward/num_tests
    return avg_reward


def run_experiment(ENV_NAME='MountainCarContinuous-v0', EPISODES=10000, TEST=10):
    """
    Runs the experiment on the target en
    """
    env = makeFilteredEnv(gym.make(ENV_NAME))

    # Create the standard DDPG agent.
    agent = DDPG(env)

    sub_critics = SubCritics(agent, order=1) # Make linear (order 1) subcritics

    # Set up tensorboard.
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('/tmp/exp1/tboard',
                                      agent.sess.graph)
    # To see graph run tensorboard --logdir=/tmp/exp1/tboard
    agent.sess.run(tf.initialize_all_variables())

    t = 0
    for episode in range(EPISODES):
        state = env.reset()
        activations = None
        print("Episode: ", episode, end="")
        r_tot = 0

        for step in range(env.spec.timestep_limit):
            t+= 1
            # Explore state space.
            next_action, next_activations = agent.noise_action_activations(state)

            # Deal with the environment
            next_state,reward,done,_ = env.step(next_action)
            r_tot += reward
            env.render()

            # Train subcrticis and plot to tensorflow
            if activations and action:
                ops, feeds = sub_critics.get_perceive_run(activations, next_activations, reward, done)
                ops += [
                    agent.critic_network.q_value_output, 
                    agent.critic_network.target_q_value_output]
                feeds.update({
                    agent.critic_network.state_input: [state],
                    agent.critic_network.action_input: [action],
                    agent.critic_network.target_state_input: [next_state],
                    agent.critic_network.target_action_input: [next_action]
                    })
                ops = [merged] + ops
                result = agent.sess.run(ops, feeds)
                train_writer.add_summary(result[0], t)

            # Train DDPG
            agent.perceive(state,next_action,reward,next_state,done)
            if done:
                break
            # Move on to next frame.
            state = next_state
            activations = next_activations
            action = next_action
        print(" ", r_tot)

        # Testing:
        if episode % 100 == 0 and episode > 100:
            avg_reward = test(env, agent, TEST)
            print(('episode: ',episode,'Evaluation Average Reward:',avg_reward))

if __name__ == '__main__':
    run_experiment()
