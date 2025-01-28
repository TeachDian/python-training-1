from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class ProbabilisticNeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.classes_ = None
        self.means_ = None
        self.priors_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.priors_ = np.array([np.mean(y == c) for c in self.classes_])
        return self

    def predict(self, X):
        probabilities = self._calculate_probabilities(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def _calculate_probabilities(self, X):
        probabilities = np.zeros((X.shape[0], len(self.classes_)))
        for idx, c in enumerate(self.classes_):
            diff = X - self.means_[idx]
            probabilities[:, idx] = self.priors_[idx] * np.exp(-np.sum(diff ** 2, axis=1) / (2 * self.sigma ** 2))
        return probabilities

    def predict_proba(self, X):
        probabilities = self._calculate_probabilities(X)
        return probabilities / probabilities.sum(axis=1, keepdims=True)