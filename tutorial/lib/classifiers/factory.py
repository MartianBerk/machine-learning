from .adaline import AdalineGD, AdalineSGD
from .linearregression import LinearRegressionGD
from .perceptron import Perceptron


class ClassifierFactory:
    @staticmethod
    def get(classifier, eta=0.1, n_iter=10, random_state=1, **kwargs):
        if classifier == 'perceptron':
            return Perceptron(eta=eta, n_iter=n_iter, random_state=random_state)
        elif classifier == 'adalinegd':
            return AdalineGD(eta=eta, n_iter=n_iter, random_state=random_state)
        elif classifier == 'adalinesgd':
            return AdalineSGD(eta=eta, n_iter=n_iter, random_state=random_state)
        elif classifier == "linearregressiongd":
            return LinearRegressionGD(eta=eta, n_iter=n_iter, random_state=random_state)
