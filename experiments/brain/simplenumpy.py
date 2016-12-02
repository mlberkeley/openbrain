import numpy as np
import gym
from common.ou_noise import OUNoise

critic_learning_rate = 10
actor_learning_rate = 100
TAU = 0.0001
GAMMA = 0.99


class Neuron:

	def __init__(self):
		self.weight = self.init_var()
		self.qa = self.init_var()
		self.qaTarget = self.init_var()
		self.weightTarget = self.init_var()
		self.t = 0
		np.random.seed(0)
	def init_var(self):
		return np.array(2* np.random.rand(1,1) - 1)

	def updateTargets(self):
		self.t += 1
		# if self.t % 10000 == 0:
		# 	self.qaTarget = self.qa
		# 	self.weightTarget = self.weight
		self.qaTarget += (1 - TAU) * (self.qa - self.qaTarget)
		self.weightTarget += (1-TAU) * (self.weight - self.weightTarget) 

	def getAction(self, state):
		return np.tanh(np.dot(self.weight, state))

	def getExplore(self, state):
		return np.random.rand([1]) * 2 - 1

	def getQ(self, action, state):
		muOffPolicy = np.tanh(np.dot(state, self.weightTarget))
		muGrads = np.dot(state, 1 - np.square(muOffPolicy))
		return np.dot(np.dot(action - muOffPolicy, muGrads), self.qa), muOffPolicy, muGrads

	def getTargetQ(self, action, state):
		muOffPolicy = np.tanh(np.dot(state, self.weightTarget))
		muGrads = np.dot(state, 1 - np.square(muOffPolicy))
		return np.dot(np.dot(action - muOffPolicy, muGrads), self.qaTarget), muOffPolicy, muGrads

	# def getQloss(self, q, qtarget, reward, done):
	# 	q, _, _ = self.getQ(action, state)
	# 	qtarget, _, _ = self.getTargetQ(nextaction, nextstate)
	# 	target = np.dot(1 - done, np.dot(GAMMA, qtarget))
		
	# 	qloss =  np.square(q - reward - target)
		
	# 	return qloss

	def trainCritic(self, reward, action, state, nextaction, nextstate, done):
		q, mu, muGrads = self.getQ(action, state)
		qtarget, muTarget, muTargetGrads = self.getTargetQ(nextaction, nextstate)
		target = np.dot(1 - done, np.dot(GAMMA, qtarget))
		grad1 = 2 * (q - reward - target)
		
		agradient = np.dot(np.dot(grad1, muGrads), action - mu)
		if agradient > 1:
			agradient = 1
		if agradient < -1:
			agradient = -1
		self.qa -= critic_learning_rate * agradient

	def trainActor(self, action, state):
		q, mu, muGrads = self.getQ(action, state)
		weightgradient = np.dot(np.dot(muGrads, muGrads), self.qa)
		if np.linalg.norm(weightgradient) > 1:
			weightgradient = 1
		if weightgradient < -1:
			weightgradient = -1
		#print("t: ",self.t, " ", self.weight)
		self.weight += actor_learning_rate * weightgradient

	def train(self, reward, action, state, nextaction, nextstate, done):
		self.updateTargets()
		self.trainCritic(reward, action, state, nextaction, nextstate, done)
		self.trainActor(action, state)

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

class ShittyMountainCar:

	def __init__(self):
		self.env = gym.make('MountainCarContinuous-v0')
		self.env.seed(0)
		self.noise = OUNoise(1)

	def featurizeState(self, state):
		##TODO
		VTOUP = 0.01
		return (state[1] - VTOUP*(state[0] + VTOUP))

	def reset(self):
		self.noise.reset()
		return self.featurizeState(self.env.reset())

	def step(self, action):
		nextstate, reward, done, _ = self.env.step(action + self.noise.noise())
		return self.featurizeState(nextstate), reward, done, _

if __name__ == '__main__':
	env = ShittyMountainCar()
	stateDim = 1#env.observation_space.shape[0]
	actionDim = 1
	layer1 = Neuron()
	layer2_1 = Neuron()
	layer2_2 = Neuron()
	layer3_1 = Neuron()
	layer3_2 = Neuron()
	layer4 = Neuron()

	EPISODES = 10000
	for episode in range(EPISODES):
		state = env.reset()
		activations = None
		print("Episode: ", episode, end="")
		r_tot = 0
		for step in range(env.env.spec.timestep_limit):
			#if episode % 100 == 0:
			env.env.render()
				
			#layer 1
			h1 = layer1.getAction(state)
			
			# #layer 2
			# h2_1 = layer2_1.getAction(h1)
			# h2_2 = layer2_2.getAction(h1)

			# #layer 3
			# h3_1 = layer3_1.getAction()
			# #output
			# action = layer3.getAction(h2_1 + h2_2)

			
			nextstate,reward,done,_ = env.step(h1)
			
			#next layer 1
			next_h1 = layer1.getAction(nextstate)

			# #next layer 2
			# next_h2_1 = layer2_1.getAction(next_h1)
			# next_h2_2 = layer2_2.getAction(next_h1)

			# #next output
			# nextaction = layer3.getAction(next_h2_1 + next_h2_2)

			#layer 1 training
			layer1.train(reward, h1, state, next_h1, nextstate, done)

			# #layer 2 training
			# layer2_1.train(reward, h2_1, h1, next_h2_1, next_h1, done)
			# layer2_2.train(reward, h2_2, h1, next_h2_2, next_h1, done)

			# #output training
			# layer3.train(reward, action, h2_1 + h2_2, nextaction, next_h2_1 + next_h2_2, done)

			r_tot += reward



			if done:
				break
			# Move on to next frame.
			state = nextstate

		print(" ", state, r_tot)
		