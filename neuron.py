import math
from enum import Enum

class ActivationFunction(Enum):
  SIGMOID = "Sigmoid"
  RELU = "Relu"
  TANH = "Tanh"
  BINARY_STEP = "Binary_step"

class Neuron:
  def __init__(self, weights, bias, func: ActivationFunction):
    self.weights = weights
    self.bias = bias
    self.func = func

  def run(self, input_data):
    if len(input_data) != len(self.weights):
      raise ValueError("The length of input_data must be equal to the length of weights")

    result = 0.0
    for i in range(len(input_data)):
      result += input_data[i] * self.weights[i]
    result += self.bias

    if self.func == ActivationFunction.SIGMOID:
      return self.__sigmoid(result)
    
    elif self.func == ActivationFunction.RELU:
      return self.__relu(result)
    
    elif self.func == ActivationFunction.TANH:
      return self.__tanh(result)
    
    elif self.func == ActivationFunction.BINARY_STEP:
      return self.__binary_step(result)
    
    else:
      raise ValueError("Unknown activation function")

  def changeBias(self, input_data):
    self.bias = input_data

  def __sigmoid(self, x):
    return 1.0 / (1.0 + math.exp(-x))

  def __relu(self, x):
    return max(0.0, x)

  def __tanh(self, x):
    return math.tanh(x)

  def __binary_step(self, x):
    return 1 if x >= 0 else 0