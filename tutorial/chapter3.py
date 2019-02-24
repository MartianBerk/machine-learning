"""
Chapter 3 - Scikit-learn.
Perceptron
Linear Regression
"""
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap

from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lib.classifiers.factory import ClassifierFactory


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='test set')


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def illustrate_single_sample():
    def cost_1(z):
        return - np.log(sigmoid(z))

    def cost_0(z):
        return - np.log(1 - sigmoid(z))

    z = np.arange(-10, 10, 0.1)
    phi_z = sigmoid(z)
    c1 = [cost_1(x) for x in z]
    plt.plot(phi_z, c1, label="J(x) if y=1")
    c0 = [cost_0(x) for x in z]
    plt.plot(phi_z, c0, linestyle="--", label="J(w) if y=0")
    plt.ylim(0.0, 5.1)
    plt.xlim([0, 1])
    plt.xlabel("$\phi$(z)")
    plt.ylabel("J(w)")
    plt.legend(loc="best")
    plt.show()


def plot_sigmoid():
    z = np.arange(-7, 7, 0.1)
    phi_z = sigmoid(z)
    plt.plot(z, phi_z)
    plt.axvline(0.0, color="k")
    plt.ylim(-0.1, 1.1)
    plt.xlabel("z")
    plt.ylabel("$\phi (z)$")

    # y axis ticks and gridline
    plt.yticks([0.0, 0.5, 1.0])
    ax = plt.gca()
    ax.yaxis.grid(True)
    plt.show()


if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=1,
                                                        stratify=y)

    # validate scikit keeps proportions in each set with stratify
    # print("Labels counts in y: {}".format(np.bincount(y)))
    # print("Labels count in y_train: {}".format(np.bincount(y_train)))
    # print("Labels counts in y_test: {}".format(np.bincount(y_test)))

    # feature scaling with StandardScaler, using same params for train and test data
    # sc = StandardScaler()
    # sc.fit(X_train)
    # X_train_std = sc.transform(X_train)
    # X_test_std = sc.transform(X_test)

    # ppn = Perceptron(n_iter=40, eta0=0.1, random_state=1)
    # ppn.fit(X_train_std, y_train)

    # y_pred = ppn.predict(X_test_std)
    # print('Misclassified samples: %d' % (y_test != y_pred).sum())
    # print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))

    # X_combined_std = np.vstack((X_train_std, X_test_std))
    # y_combined = np.hstack((y_train, y_test))
    # plot_decision_regions(X_combined_std, y_combined, ppn, test_idx=range(105, 150))
    # plt.xlabel("petal length [standardized]")
    # plt.ylabel("petal width [standardized]")
    # plt.legend(loc="upper left")
    # plt.show()

    # plot_sigmoid()
    # illustrate_single_sample()

    # Demo linear regression, using classes 0 and 1 as only binary classification possible
    X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
    lrgd = ClassifierFactory.get("linearregressiongd", eta=0.05, n_iter=1000, random_state=1)
    lrgd.fit(X_train_01_subset, y_train_01_subset)
    plot_decision_regions(X_train_01_subset, y_train_01_subset, classifier=lrgd)
    plt.xlabel("petal length [standardized]")
    plt.ylabel("petal width [standardized]")
    plt.legend(loc="upper left")
    plt.show()/

