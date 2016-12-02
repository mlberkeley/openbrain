"""
A numpy implementation of various activations.
"""
import numpy as np

from abc import ABC, abstractmethod

class Activation(ABC):
	"""
	Stores various activation functions implemented in numpy
	and their shape invariant derivatives.
	"""
	def __call__(self, tensor):
		"""
		Evaluates f(TENSOR) where f is the activation.
		"""
		return self.func(tensor)

	def __invert__(self):
		"""
		Evaluates the derivaitve of the actiavtion
		applied at tensor. Eg,
		(~f)(x) = f'(x)
		"""
		return self.derivative


	@abstractmethod
	def func(self, tensor):
		"""
		Evalautes the activation.
		"""
		return NotImplemented


	@abstractmethod
	def derivative(self, tensor):
		"""
		Evalautes the symbolic derivative of the 
		activation.
		"""
		return NotImplemented



class Tanh(Activation):
	"""
	&&&&&
	"""
	def func(self, tensor):
		return np.tanh(tensor)

	def derivative(self, tensor):
		return 1- np.square(np.tanh(tensor))

class Relu(Activation):
	"""
	&&&&&
	"""
	def func(self, tensor):
		return np.maximum(0, tensor)

	def derivative(self, tensor):
		return (tensor > 0)*1