import math

class NaiveBayes:
  def __init__(self):
    self.classes = None
    self.priors = None
    self.features_by_class = None

  def fit(self, X, y):
    """
    Train the Naive Bayes model on the given data.

    Args:
      X: A list of lists, where each inner list represents a data point and its features.
      y: A list of class labels corresponding to each data point in X.
    """
    self.classes = list(set(y))
    self.priors = {cls: y.count(cls) / len(y) for cls in self.classes}
    self.features_by_class = self._summarize_by_class(X, y)

  def predict(self, X):
    """
    Predict the class labels for a list of data points.

    Args:
      X: A list of lists, where each inner list represents a data point and its features.

    Returns:
      A list of predicted class labels.
    """
    predictions = []
    for x in X:
      posteriors = {cls: self._calculate_posterior(cls, x) for cls in self.classes}
      predicted_class = max(posteriors, key=posteriors.get)
      predictions.append(predicted_class)
    return predictions

  def _summarize_by_class(self, X, y):
    """
    Summarize the data by class, calculating means and standard deviations for continuous features
    and frequency tables for discrete features.

    Args:
      X: A list of lists, where each inner list represents a data point and its features.
      y: A list of class labels corresponding to each data point in X.

    Returns:
      A dictionary where keys are class labels and values are dictionaries containing summary statistics for each feature.
    """
    features_by_class = {cls: {} for cls in self.classes}
    for x, label in zip(X, y):
      for i, feature in enumerate(x):
        if feature not in features_by_class[label]:
          if isinstance(feature, (int, float)):
            features_by_class[label][i] = {"mean": 0, "std": 0, "count": 0}
          else:
            features_by_class[label][i] = {"counts": {}}
        if isinstance(feature, (int, float)):
          features_by_class[label][i]["mean"] += feature
          features_by_class[label][i]["std"] += feature**2
          features_by_class[label][i]["count"] += 1
        else:
          features_by_class[label][i]["counts"][feature] = features_by_class[label][i]["counts"].get(feature, 0) + 1
    for cls in self.classes:
      for feature, stats in features_by_class[cls].items():
        if stats.get("count"):
          stats["mean"] /= stats["count"]
          stats["std"] = math.sqrt(stats["std"] / stats["count"] - stats["mean"]**2)
    return features_by_class

  def _calculate_posterior(self, cls, x):
    """
    Calculate the posterior probability of a class given a data point using Bayes' theorem.

    Args:
      cls: The class label.
      x: A list representing a data point.

    Returns:
      The posterior probability of the class given the data point.
    """
    posterior = self.priors[cls]
    for i, feature in enumerate(x):
      if isinstance(feature, (int, float)):
        posterior *= self._gaussian_probability(feature, self.features_by_class[cls][i])
      else:
        posterior *= self._discrete_probability(feature, self.features_by_class[cls][i])
    return posterior

  def _gaussian_probability(self, x, stats):
    """
    Calculate the probability of a continuous feature value given a class using the Gaussian probability density function.

    Args:
      x: The feature value.
      stats: A dictionary containing mean and standard deviation of the feature for the class.

    Returns:
      The probability of the feature value given the class.
    """
    if stats["std"] == 0:
      return 1  
