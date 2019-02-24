from .adaline import AdalineGD, AdalineSGD
from .perceptron import Perceptron


class RuleFactory:
    @staticmethod
    def get(rule, eta=0.1, n_iter=10, random_state=1, **kwargs):
        if rule == 'perceptron':
            return Perceptron(eta=eta, n_iter=n_iter, random_state=random_state)
        elif rule == 'adalinegd':
            return AdalineGD(eta=eta, n_iter=n_iter, random_state=random_state)
        elif rule == 'adalinesgd':
            return AdalineSGD(eta=eta, n_iter=n_iter, random_state=random_state)
