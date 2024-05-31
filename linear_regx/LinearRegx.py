import numpy as np

class LinearRegression:
  """
  A class implementing linear regression with gradient descent.
  """

  def __init__(self, learning_rate=0.01, num_iters=1000):
    """
    Initializes the LinearRegression class.

    Args:
      learning_rate: The learning rate for gradient descent (default: 0.01).
      num_iters: The number of iterations for gradient descent (default: 1000).
    """
    self.learning_rate = learning_rate
    self.num_iters = num_iters
    self.weights = None
    self.bias = None



