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
      
  def fit(self, X, y):
      """
      Fits the model to the training data.

      Args:
        X: A numpy array of shape (m, n) representing the features (m data points, n features).
        y: A numpy array of shape (m,) representing the target values.
      """
      # Add a column of ones to X for the bias term
      X = np.hstack((np.ones((X.shape[0], 1)), X))
      self.weights = np.zeros(X.shape[1])
      self.bias = 0

      # Gradient descent loop
      for _ in range(self.num_iters):
        y_predicted = self.predict(X)
        errors = y - y_predicted
        self.weights -= self.learning_rate * np.dot(X.T, errors)
        self.bias -= self.learning_rate * np.mean(errors)

  def predict(self, X):
    """
    Predicts the target values for new data points.

    Args:
      X: A numpy array of shape (m, n) representing the features of new data points.

    Returns:
      A numpy array of shape (m,) representing the predicted target values.
    """
    # Add a column of ones to X for the bias term
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    return np.dot(X, self.weights) + self.bias
