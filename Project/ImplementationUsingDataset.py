import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            # approximate output variable (y) with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))  # derivative w.r.t weights
            db = (1 / n_samples) * np.sum(y_predicted - y)  # derivative w.r.t bias
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# === Separate training/testing logic ===
def run_logistic_regression():
    itr = []
    acc = []

    d = datasets.load_breast_cancer()
    x, y = d.data, d.target
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=1234)

    regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    regressor.fit(xtrain, ytrain)

    regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    regressor.fit(xtrain, ytrain)
    predictions = regressor.predict(xtest)
    itr.append(1000)

    print("LR classification accuracy:", accuracy(ytest, predictions))
    acc.append(accuracy(ytest, predictions))

    print(regressor.weights)

    print(regressor.bias)

    plt.scatter(itr, acc, color="r")
    plt.plot(itr, acc)
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.show()


run_logistic_regression()
