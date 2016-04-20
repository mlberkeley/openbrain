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
import scipy.special import expit

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
    def __init__(self, evalFn="scoreEvaluation", num_neurons=1000 ):
        self.num_inputs = 100 #TODO make sure corrrect.
        self.num_outputs = 4
        total_neurons = num_neurons+num_inputs+num_outputs
        self.W = np.random.random((total_neurons, total_neurons))
        self.W *= 1 - np.eye((total_neurons, total_neurons)) #delete self connections
        self.W *= 1- np.outer( (np.arange(total_neurons)>=(total_neurons-num_outputs))*1, (np.arange(total_neurons)>=(total_neurons-num_outputs))*1)
        self.decay_const = 0.8

        #Todo gaussian
        self.rho = np.zeros((total_neurons,))
        self.rho_naught = np.ones((total_neurons,))
        self.rho_naught *= (np.arange(total_neurons) > )*1
        self.R = np.ones((total_neurons,))
        self.v = np.zeros((total_neurons,)) #Voltages
        self.activation = expit

    def get_outputs(self):
        return self.v[:-self.num_outputs]
    def set_inputs(self, i):
        self.v[:self.num_inputs] = i #TODO stochastic. 
    
    def get_inputs(self, state):
        pass

    def update(self,input_state):
        self.v *= self.decay_const
        self.set_inputs(input_state))
        delta_v = self.activation(W.dot(R*self.v))
        self.v = self.v + delta_v - (self.v*self.R)
        self.rho -= (1*(self.rho != 0)) - self.rho_naught*self.R
        self.R = (self.rho == 0)*1

    def learn(self,state):
        pass

    def getAction(self, state):
        incoming_inputs = self.get_inputs(state)
        self.update(self,incoming_inputs)
        self.learn(self,state)
        out_vect = self.get_outputs()
        output = fhoaifhas (out_vect)
        return 

        # Generate candidate actions
        # legal = state.getLegalPacmanActions()
        # if Directions.STOP in legal: legal.remove(Directions.STOP)

        # successors = [(state.generateSuccessor(0, action), action) for action in legal]
        # scored = [(self.evaluationFunction(state), action) for state, action in successors]
        # bestScore = max(scored)[0]
        # bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random.choice(bestActions)
