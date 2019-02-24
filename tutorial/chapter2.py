"""
Chapter 2 - Classification Algorithms.
Perceptron
Adaline - Gradient Descent
Adaline - Stochastic Gradient Descent
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from lib.rules.rulefactory import RuleFactory


def plot_data(X):
    # plot extracted data to analyze
    
    plt.scatter(X[:50, 0], X[:50, 1],
                color="red", marker="o", label="setosa")
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color="blue", marker="x", label="versicolor")
    plt.xlabel("sepal length (cm)")
    plt.ylabel("petal length (cm)")
    plt.legend(loc="upper left")
    plt.show()


def plot_misclassification(rule, rule_attr, method):
    # plot the misclassifications from Perceptron to show when convergance happened

    plt.plot(range(1, len(getattr(rule, rule_attr)) + 1), getattr(rule, rule_attr), marker="o")
    plt.xlabel("Epochs")
    plt.ylabel(method)
    plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # plot decision boundaries for 2D datasets

    # setup marker generator and color map
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor="black")


def learn_perceptron(X, y, eta, n_iter, show_learn=False, show_decision_regions=False):
    pct = RuleFactory.get('perceptron', eta=eta, n_iter=n_iter)
    pct.fit(X, y)

    if show_learn:
        # plot misclassifcation error for each epoch
        plot_misclassification(pct, "errors_", "Number of Updates")

    if show_decision_regions:
        # plot decision regions
        plot_decision_regions(X, y, classifier=pct)
        plt.xlabel("sepal length (cm)")
        plt.ylabel("petal length (cm)")
        plt.legend(loc="upper left")
        plt.show()

    return pct


def learn_adaline(X, y, eta, n_iter, show_learn=False, show_decision_regions=False):
    ada = RuleFactory.get("adalinegd", eta=eta, n_iter=n_iter)
    ada.fit(X, y)

    if show_learn:
        plot_misclassification(ada, "cost_", "Sum-squared-error")

    if show_decision_regions:
        plot_decision_regions(X, y, classifier=ada)
        plt.xlabel("sepal length [standardized]")
        plt.ylabel("petal length [standardized]")
        plt.title("Adaline - Gradient Descent")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    return ada


def learn_adaline_sgd(X, y, eta, n_iter, show_learn=False, show_decision_regions=False):
    ada = RuleFactory.get("adalinesgd", eta=eta, n_iter=n_iter)
    ada.fit(X, y)

    if show_learn:
        plot_misclassification(ada, "cost_", "Average Cost")

    if show_decision_regions:
        plot_decision_regions(X, y, classifier=ada)
        plt.title("Adaline - Stochastic Gradient Descent")
        plt.xlabel("sepal length [standardized]")
        plt.ylabel("petal length [standardized]")
        plt.legend(loc="upper left")
        plt.show()

    return ada


if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    local_file = "source/iris.csv"
    df = pd.read_csv(local_file, header=None)

    # select setosa and versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", -1, 1)

    # extract sepal length and patel length
    X = df.iloc[0:100, [0, 2]].values

    # plot extracted data
    # plot_data(X)

    # train Perceptron with extracted data
    # learn_perceptron(X, y, 0.1, 10)

    # train Adaline with extracted data
    # ada1 = learn_adaline(X, y, 0.01, 10)
    # ada2 = learn_adaline(X, y, 0.0001, 10)

    # plot cost against learning rates, showing problems with choosing
    # the wrong learning rate
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # ax[0].plot(range(1, len(ada1.cost_) + 1),
    #            np.log10(ada1.cost_),
    #            marker="o")
    # ax[0].set_xlabel("Epochs")
    # ax[0].set_ylabel("log(Sum-squared-error)")
    # ax[0].set_title("Adaline - Learning Rate 0.01\n(doesn't converge, overshoots global minimum)")

    # ax[1].plot(range(1, len(ada2.cost_) + 1),
    #            ada2.cost_,
    #            marker="o")
    # ax[1].set_xlabel("Epochs")
    # ax[1].set_ylabel("Sum-sqaured-error")
    # ax[1].set_title("Adaline - Learning Rate 0.0001\n(takes too long to converge)")

    # use standardization as a feature scaling method to improve performance
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    # learn_adaline(X_std, y, 0.01, 15, show_learn=True, show_decision_regions=True)

    # use stochastic gradient Adaline algorithm with standardized dataset
    learn_adaline_sgd(X_std, y, 0.01, 15, show_learn=True, show_decision_regions=True)
