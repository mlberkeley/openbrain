import numpy as np
import gym
from common.ou_noise import OUNoise
import common.activations as activations
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
CRITIC_LR = 10
ACTOR_LR = 3
VALUE_LR = 10
TAU = 0.01
BIAS = [1]
CLIP = 2
GAMMA = 0.9


class Neuron:

	def __init__(self, input_shape, output_shape = [1], activation=None):
		self.activation = activations.Tanh() if not activation else activation

		self.in_shape = input_shape if type(input_shape) is list else [input_shape]
		self.out_shape = output_shape if type(output_shape) is list else [output_shape]
		
		self.in_shape = self.in_shape[:-1] + [self.in_shape[-1] + 1]
		self.actor_weights = self.init_var(self.in_shape)
		self.critic_weights = self.init_var(self.in_shape)
		self.value_weights = self.init_var(self.in_shape)

		self.critic_weightsDelay = self.init_var(self.in_shape )
		self.actor_weightsDelay = self.init_var(self.in_shape)
		self.value_weightsDelay = self.init_var(self.in_shape)
		self.t = 0
		np.random.seed(0)


	def init_var(self, shape = [1]):
		"""
		Uniform initialization code.
		"""
		return np.array(2* np.random.rand(*shape) - 1)*1e-1

	def updateDelays(self):
		"""
		Weight transfer.
		"""
		self.t += 1
		# if self.t % 10000 == 0:
		# 	self.critic_weightsDelay = self.critic_weights
		# 	self.actor_weightsDelay = self.actor_weights
		self.critic_weightsDelay += (1 - TAU) * (self.critic_weights - self.critic_weightsDelay)
		self.actor_weightsDelay += (1-TAU) * (self.actor_weights - self.actor_weightsDelay) 
		self.value_weightsDelay += (1 - TAU) * (self.value_weights - self.value_weightsDelay)

		# Is this really an EMA?

	def getAction(self, state):
		"""
		Gets the current action of the neuron.
		Assumes state is a vector in R^d.
		"""
		state = np.append(state,[1])
		return self.activation(np.dot(self.actor_weights, state))

	def getDelayAction(self, state):
		"""
		Gets the Delay action of the neuron.
		Assumes state is a vector in R^d.
		"""
		state = np.append(state,[1])
		return self.activation(np.dot(self.actor_weightsDelay,state))

	def getQ(self, action, state):
		"""
		Gets Q with respect to the behaviour policy. 
		"""
		bstate = np.append(state,[1])
		return self._calcCompatibleQ(action, bstate, delay=False)

	def getDelayQ(self, action, state):
		"""
		Gets Q with respect to the lag policy (weightsTarget) as the target policy.
		"""

		bstate = np.append(state,[1])
		return self._calcCompatibleQ(action, bstate, delay=True)

	def _calcCompatibleQ(self, action, bstate, delay=False):
		"""
		Calculates the compatible Q.
		
		Note: The DPG paper does not have the notion of a delay Q, and thus 
		there is not an exact reconciliation with DDPG. Our implementation
		uses: Q^delay(s,a) ~ (a - mu^delay(s)) .. 
		"""
		act_weights = self.actor_weightsDelay if delay else self.actor_weights
		crit_weights = self.critic_weightsDelay if delay else self.critic_weights
		val_weights = self.value_weightsDelay if delay else self.value_weights

		# internal calc
		net = np.dot(self.actor_weights,bstate)

		muAction = self.activation(net)
		# ~ is the derivative syntax
		Dmu_Dnet = (~self.activation)(net)
		Dmu_Dweights = np.dot(Dmu_Dnet, bstate.T)

		deviation = (action - muAction)*Dmu_Dweights
		V = np.dot(self.value_weights, bstate)
		Q = np.dot(deviation, self.critic_weights) + V

		return  Q, deviation, Dmu_Dweights

	def trainCritic(self, reward, action, state, nextstate, done):
		"""
		Trains the critic using a DPG compatable update.
		"""

		bstate = np.append(state,[1])
		bnextstate = np.append(nextstate,[1])
		# Calling Q target is misleading. It is a stable lag network.
		Q, muDeviation, _ = self.getQ(action, state)
		QDelay, _, _ = self.getDelayQ(self.getDelayAction(nextstate), nextstate)
		value  = np.dot(1 - done, np.dot(GAMMA, QDelay))

		# Compute the derivative of the loss w.r.t the q weights

		delta = (Q - reward - value)
		DQ_Dcrit_weights = delta*muDeviation
		DQ_Dval_weights = delta*(bstate)

		if self.t % 1000 == 0 or done:
			print("Q_LOSS", delta**2, "Q", Q, "QDelay", QDelay)
		

		self.critic_weights -= CRITIC_LR * self._clipGradient(DQ_Dcrit_weights)
		self.value_weights -= VALUE_LR * self._clipGradient(DQ_Dval_weights)

	def trainActor(self, action, state):
		Q, muDeviation, muGrads = self.getQ(action, state)
		bstate = np.append(state,[1])
		bnextstate = np.append(nextstate,[1])
		DQ_Dact_weights = np.dot(np.dot(muGrads, muGrads), self.critic_weights)

		#print("t: ",self.t, " ", self.actor_weights)
		self.actor_weights += ACTOR_LR * self._clipGradient(DQ_Dact_weights)


	def _clipGradient(self, grad):

		if np.linalg.norm(grad) > CLIP:
			return np.dot(grad, CLIP/np.linalg.norm(grad))
		else:
			return grad

	def train(self, reward, action, state, nextstate, done):
		self.updateDelays()
		self.trainCritic(reward, action, state, nextstate, done)
		self.trainActor(action, state)

	def info(self):
		print(self.critic_weights)
		print(self.value_weights)
		print(self.actor_weights)
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

	def featurizeState(self, state):
		##TODO
		VTOUP = 0.01
		return (state[1] - VTOUP*(state[0] + VTOUP))

	def reset(self):
		return self.featurizeState(self.env.reset())

	def step(self, action):
		nextstate, reward, done, _ = self.env.step(action)
		return self.featurizeState(nextstate), reward, done, _



def qplot(env, layer, name): 
	minPos = -1.2
	maxPos = 0.6
	minVelocity = -0.07
	maxVelocity = 0.07
	nx = 100
	ny = 100
	num_surface = 10
	xRange = np.linspace(minPos, maxPos, nx)
	yRange = np.linspace(minVelocity, maxVelocity, ny)
	xv, yv = np.meshgrid(xRange, yRange)
	zToPlot = []
	xToPlot = xv[0]
	yToPlot = [row[0] for row in yv]
	for i in range(nx):
		for j in range(ny):
			x, y = xv[i,j], yv[i,j]
			zToPlot += [np.argmax([layer.getQ((d)/num_surface - 1/2, [x,y])[0]
											 for d in range(num_surface)])/num_surface - 1/2]
			#print(evals)
			#result = np.argmax(evals)/10 -0.5
			#zToPlot.append(evals[0])


	zToPlot = np.array(zToPlot)

	fig = plt.figure()	
	for i in range(num_surface):
		zToPlots = np.array(zToPlot).reshape((len(xToPlot), len(yToPlot)))

		ax = fig.gca(projection='3d')
		if i == 1:
			h = ax.plot_surface(xv,yv,zToPlots,cmap=cm.jet)
		plt.title('Policy')
		plt.xlabel('position')
		plt.ylabel('action')
	
	fig.colorbar(h, shrink=0.5, aspect=5)
	fig.savefig(name)

import os, shutil
folder = './figs'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

env = gym.make('MountainCarContinuous-v0')
stateDim = 2#env.observation_space.shape[0]
actionDim = 1
layer1 = Neuron(stateDim)

noise = OUNoise(actionDim)

EPISODES = 200000
for episode in range(EPISODES):
	state = env.reset()
	noise.reset()
	print("Episode: ", episode, end="")
	r_tot = 0
	if episode % 10 == 0 or episode < 10:
		layer1.info()
		qplot(env, layer1, "./figs/ep{}.png".format(episode))

		
	for step in range(env.spec.timestep_limit):
		if episode % 1000 == 0 and episode > 0:
			env.render()

		act = layer1.getAction(state) + noise.noise()
		
		nextstate,reward,done,_ = env.step(act)
		layer1.train(reward, act, state, nextstate, done)

		r_tot += reward
		


		if done:
			break
		# Move on to next frame.
		state = nextstate

	print(" ", state, r_tot)
