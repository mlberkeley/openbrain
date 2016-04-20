# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
import random
import game
import util
import numpy as np
from  scipy.special import expit
import networkx as nx
import matplotlib.pyplot as plt
import math

class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        

        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP: current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal: return left
        if current in legal: return current
        if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal: return Directions.LEFT[left]
        return Directions.STOP

class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal: legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        scored = [(self.evaluationFunction(state), action) for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)

def scoreEvaluation(state):
    return state.getScore()


class OpenBrainAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation", num_neurons=200 ):
        self.num_inputs = 1 #TODO make sure corrrect.
        self.num_outputs = 4
        self.total_neurons = num_neurons + self.num_inputs + self.num_outputs
        self.W = (np.random.random((self.total_neurons, self.total_neurons)) > (1 - 0.5*(1/math.sqrt(num_neurons))))*np.random.random((self.total_neurons, self.total_neurons)) 
        self.W *= 1 - np.eye(self.total_neurons) #delete self connections
        self.decay_const = 0.5 

        #Todo gaussian
        self.rho = np.zeros((self.total_neurons,))
        self.rho_naught = np.ones((self.total_neurons,))*2
        self.rho_naught *= 1- (np.arange(self.total_neurons) >= (self.total_neurons-self.num_outputs))*1
        self.R = np.ones((self.total_neurons,))
        self.v = np.zeros((self.total_neurons,)) #Voltages
        self.activation = expit

        self.threshold =1

        self.visualize()

    def get_outputs(self):
        retr = np.copy(self.v[-self.num_outputs:])
        self.v[-self.num_outputs:] = 0
        return retr

    def visualize(self):
        self.G = nx.from_numpy_matrix(self.W,create_using=nx.MultiDiGraph())
        nx.draw(self.G, cmap = plt.get_cmap('jet'))
        plt.ion()
        plt.show()

    def set_inputs(self, i):
        self.v[:self.num_inputs] = i #TODO stochastic. 
    
    def get_inputs(self, state):
        return 0.1

    def update(self,input_state):
        self.v *= self.decay_const
        self.set_inputs(input_state)
        delta_v = self.W.dot(self.R*self.activation(self.v))
        self.v = self.v - (self.v*self.R)
        self.v = self.v + delta_v 
        self.rho = self.rho -1  + self.rho_naught*self.R
        self.R = ((self.rho <= 0)*1) * (( self.v > self.threshold)*1)

    def learn(self,state):
        pass

    def output_to_action(self, outputs):
        choice = np.random.choice(np.argwhere(outputs == np.amax(outputs)).ravel())
        if outputs[choice] >= 0.5:
            if choice == 0:
                return Directions.WEST
            elif choice == 1:
                return Directions.EAST
            elif choice == 2:
                return Directions.NORTH
            else:
                return Directions.SOUTH
        else: 
            return Directions.STOP

    def getAction(self, state):
        incoming_inputs = self.get_inputs(state)
        self.update(incoming_inputs)
        self.learn(state)
        print(self.v[-self.num_outputs:])
        out_vect = self.get_outputs()
        #get output actions from out_vect
        action = self.output_to_action(out_vect)
        print(action)
        if action not in  state.getLegalPacmanActions():
            return Directions.STOP
        else:
            return action

# Generate candidate actions
# legal = state.getLegalPacmanActions()
# if Directions.STOP in legal: legal.remove(Directions.STOP)

# successors = [(state.generateSuccessor(0, action), action) for action in legal]
# scored = [(self.evaluationFunction(state), action) for state, action in successors]
# bestScore = max(scored)[0]
# bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
# return random.choice(bestActions)
