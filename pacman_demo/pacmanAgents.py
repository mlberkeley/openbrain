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
import IPython

def scoreEvaluation(state):
    return state.getScore()

class _OpenBrainInput():

    def __init__(self):
        self.n = None
        self.m = None
        self.input_vec = None
        self._last_pacman_coord = None
        self._last_ghost_coords = None
        self._last_foods = None

    def get_vec(self):
        return self.input_vec.copy()

    @staticmethod
    def _grid_to_matrix(grid):
        n, m = grid.height, grid.width
        matrix = np.zeros([n, m])
        for x, y in grid.asList():
            matrix[n-y-1][x] = 1
        return matrix

    def _pos_to_coord(self, pos):
        return self.n - pos[1], pos[0]

    def _init_input_vec(self, state):
        walls_mat = _OpenBrainInput._grid_to_matrix(state.getWalls())
        foods_mat = _OpenBrainInput._grid_to_matrix(state.getFood())

        self.n, self.m = walls_mat.shape

        pacman_mat = np.zeros([self.n ,self.m])
        pacman_pos = state.getPacmanPosition()
        self._last_pacman_coord = self._pos_to_coord(pacman_pos)
        pacman_mat[self._last_pacman_coord] = 1

        ghosts_mat = np.zeros([self.n, self.m])
        ghost_poss = state.getGhostPositions()
        self._last_ghost_coords = [self._pos_to_coord(pos) for pos in ghost_poss]
        for coord in self._last_ghost_coords:
            ghosts_mat[coord] = 1

        large_mat = np.r_[walls_mat, np.r_[foods_mat, np.r_[pacman_mat, ghosts_mat]]]
        self.input_vec = large_mat.ravel()

        self._last_foods = state.getFood().copy()

    def _flat_coord(self, k, coord):
        return self.n * self.m * k + self.m * coord[0] + coord[1]

    def update(self, state):
        if self.input_vec is None:
            self._init_input_vec(state)
        else:
            pacman_pos = state.getPacmanPosition()
            pacman_coord = self._pos_to_coord(pacman_pos)

            #update foods
            if self._last_foods[pacman_pos[0]][pacman_pos[1]]:
                self.input_vec[self._flat_coord(1, pacman_coord)] = 0
                self._last_foods = state.getFood().copy()

            #update pacman
            self.input_vec[self._flat_coord(2, self._last_pacman_coord)] = 0
            self.input_vec[self._flat_coord(2, pacman_coord)] = 1
            self._last_pacman_coord = pacman_coord

            #update ghosts
            for ghost_coord in self._last_ghost_coords:
                self.input_vec[self._flat_coord(3, ghost_coord)] = 0
            self._last_ghost_coords = []
            for ghost_pos in state.getGhostPositions():
                coord = self._pos_to_coord(ghost_pos)
                self._last_ghost_coords.append(coord)
                self.input_vec[self._flat_coord(3, coord)] = 1

#CONSTANTS
#TODO: PUT IN PARAM FILES
DENSITY = 0.4
RHO_0 = 2

class OpenBrainAgent(Agent):

    def __init__(self, evalFn="scoreEvaluation", num_neurons=200 ):
        self.num_inputs = 880 #TODO make sure corrrect.
        self.num_outputs = 4
        self.total_neurons = num_neurons + self.num_inputs + self.num_outputs
        self.decay_const = 0.5

        #init connections
        #make random connections
        self.W = np.random.random((self.total_neurons, self.total_neurons)) > (1 - DENSITY/math.sqrt(num_neurons))
        #give random values to initial weights
        self.W = self.W*np.random.random((self.total_neurons, self.total_neurons))
        self.W *= 1 - np.eye(self.total_neurons) #delete self connections

        #Todo gaussian
        self.rho = np.zeros((self.total_neurons,))
        self.rho_naught = np.ones((self.total_neurons,))*RHO_0
        self.rho_naught *= 1- (np.arange(self.total_neurons) >= (self.total_neurons-self.num_outputs))*1
        self.R = np.ones((self.total_neurons,))
        self.v = np.zeros((self.total_neurons,)) #Voltages
        self.activation = expit

        self.threshold =1
        self.rho_reinforce = 10
        # self.visualize()

        #state vars
        self._input = _OpenBrainInput()
        self._last_state = None

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
        self._input.update(state)
        return self._input.get_vec()

    def update(self,input_state):
        self.v *= self.decay_const

        self.set_inputs(input_state)

        delta_v = self.W.T.dot(self.R * self.activation(self.v))
        self.v = self.v - (self.v * self.R)
        self.v = self.v + delta_v
        self.rho = self.rho -1  + self.rho_naught*self.R
        self.R = ((self.rho <= 0)*1) * (( self.v > self.threshold)*1)

    def _get_rewards(self, state):
        rewards = {}
        rewards['food'] = self._last_state.getNumFood() - state.getNumFood()
        rewards['points'] = state.getScore() - self._last_state.getScore()

        return rewards

    def learn(self,state):
        if self._last_state is None:
            return

        rewards = self._get_rewards(state)

        #if positive reward
        #reinforce connections of all neurons that are in refractory period
        for name, reward in rewards.items():
            if name == 'food' and reward > 0:
                targets = self.rho > 0
                #need to multiply weights of all connections of all neurons in targets
                #multiplier is (1 + inverse rho * scale)
                multiplier = 1 + self.rho_reinforce * np.where(targets, 1./self.rho, 0)
                self.W = (self.W.T * multiplier).T
            if name == 'points':
                if reward > 0:
                    targets = self.rho > 0
                    #need to multiply weights of all connections of all neurons in targets
                    #multiplier is (1 + inverse rho * scale)
                    multiplier = 1 + self.rho_reinforce * np.where(targets, 1./self.rho, 0)
                    # print 'multiplier', multiplier
                    self.W = (self.W.T * multiplier).T
                else:
                    targets = self.rho < 0
                    #need to multiply weights of all connections of all neurons in targets
                    #multiplier is (1 + inverse rho * scale)
                    multiplier = 1 - self.rho_reinforce * np.where(targets, 1./self.rho, 0)
                    # print 'multiplier', multiplier
                    self.W = (self.W.T * multiplier).T
                self.W = self.W * multiplier.dot(multiplier.T)

    def output_to_action(self, outputs):
        satisfied_neurons = np.argwhere(outputs >= self.threshold)
        n = len(satisfied_neurons)
        if n == 0:
            return Directions.STOP

        choice = satisfied_neurons[int(np.random.rand() * n)]
        if choice == 0:
            return Directions.WEST
        elif choice == 1:
            return Directions.EAST
        elif choice == 2:
            return Directions.NORTH
        else:
            return Directions.SOUTH

    def getAction(self, state):
        incoming_inputs = self.get_inputs(state)
        self.update(incoming_inputs)
        self.learn(state)
        self._last_state = state.deepCopy()

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
