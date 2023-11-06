import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    
    def __init__(self, n_estimators, learning_rate):
        self.__n_estimators = n_estimators
        self.__learning_rate = learning_rate
        self.__m = 0
        self.__b = 0

    def __loss_function(self, m, b, points, keys):
        total_error = 0
        for i in range(len(points)):
            x = points[i].iloc[i][keys[0]]
            y = points[i].iloc[i][keys[1]]
            total_error += (y - (m * x + b)) ** 2
        return total_error / float(len(points))

    
    def __gradient_descent(self, m_now, b_now, points):
        m_gradient = 0
        b_gradient = 0

        n = len(points)
        for i in range(n):
            x = points.iloc[i][self.__keys[0]]
            y = points.iloc[i][self.__keys[1]]

            m_gradient += - (2/n) * x * (y - (m_now * x + b_now))
            b_gradient += - (2/n) * (y - (m_now * x + b_now))

        m = m_now - m_gradient * self.__learning_rate
        b = b_now - b_gradient * self.__learning_rate

        return m, b
    

    def fit(self, X, y, column_index):
        points = pd.concat([X, y], axis=1)
        self.__keys = [X.columns[column_index], y.name]
        
        for i in range(self.__n_estimators):
            self.__m, self.__b = self.__gradient_descent(self.__m, self.__b, points)
        
        return self.__m, self.__b


    def predict(self, X):
        return X.apply(lambda row: self.__m * row[self.__keys[0]] + self.__b, axis='columns')