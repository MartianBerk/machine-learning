import numpy as np

class BaseRule:
    """Base rule to be inherited by Rules."""

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def fit(self, X, y):
        raise NotImplementedError
