from collections import deque
import numpy as np
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size, scale_param = 0.5, uniform = False):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()
        self.uniform = uniform

    def get_batch(self, batch_size):
        if self.uniform:
            return random.sample(list(self.buffer), batch_size)
        scale = self.scale_param * len(self.buffer)
        inds = np.rint(np.random.exponential(scale = scale, size = batch_size))
        # only go as far back as the start of the deque...
        inds[inds > len(self.buffer)] = len(self.buffer)
        # start from the end of the deque
        inds = -inds
        return [self.buffer[int(i)] for i in inds]

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0
