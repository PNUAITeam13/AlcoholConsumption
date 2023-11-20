import numpy as np

class LinearRegressionMultipleFeatures:
    def __init__(self, n_estimators, learning_rate):
        self.__n_estimators = n_estimators
        self.__learning_rate = learning_rate

    def gradient_descent(self, X, y):
        predictions = self.predict(X)
        errors = predictions - y.reshape(-1)

        self.__m -= self.__learning_rate * (1 / X.shape[0]) * X.T.dot(errors)
        self.__b -= self.__learning_rate * (1 / X.shape[0]) * np.sum(errors)

    def loss_function(self, X, y):
        predictions = self.predict(X)
        errors = predictions - y
        loss = np.mean(errors**2) / 2
        return loss

    def fit(self, X, y):
        X = X.values
        y = y.values.reshape(-1, 1)

        self.__m = np.zeros(X.shape[1])
        self.__b = 0
        self.__loss_history = []

        for i in range(self.__n_estimators):
            self.gradient_descent(X, y)

            current_loss = self.loss_function(X, y)
            self.__loss_history.append(current_loss)

    def predict(self, X):
        return (np.dot(X, self.__m) + self.__b).flatten()