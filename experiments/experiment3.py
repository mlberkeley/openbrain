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

VTOUP = 0.001

from brain.common.filter_env import makeFilteredEnv
from brain.global_brain import GlobalBrain

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
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.50
    env = gym.make(ENV_NAME)
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    sess = tf.Session(config=config)
    brain = GlobalBrain(sess, 1, actionDim)

    # Set up tensorboard.
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('/tmp/tboard/{}'.format(exp_name),
                                      sess.graph)
    # To see graph run tensorboard --logdir=/tmp/exp1/tboard
    sess.run(tf.initialize_all_variables())
    w = 0 
    for episode in range(EPISODES):
        state = env.reset()
        activations = None
        print("Episode: ", episode, end="")
        r_tot = 0
        t = 0

        for step in range(100):
            w += 1
            t+=1
            action = brain.getAction(state[1] - VTOUP*(state[0]) + VTOUP)
            # Deal with the environment
            
            #print(action)
            env.render()
           
            next_state,reward,done, _ = env.step(action)

            r_tot += reward
            ops, feeds = brain.perceive(reward, int(done), (state[1] - VTOUP*(state[0]) + VTOUP), action, next_state[1] - VTOUP*(next_state[0]) + VTOUP)
            if ops != None and feeds != None:
                ops = [merged] + ops
                result = sess.run(ops, feeds)
                train_writer.add_summary(result[0], w)

            if done:
                break
            # Move on to next frame.
            state = next_state
        print(" ", r_tot)

        # Testing:
        # if episode % 100 == 0 and episode > 100:
            #avg_reward = test(env, brain, TEST)
            #print(('episode: ',episode,'Evaluation Average Reward:',avg_reward))

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
