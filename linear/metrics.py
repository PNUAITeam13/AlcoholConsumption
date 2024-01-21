import numpy as np
from functools import wraps


class MetricsWrapper:
    @staticmethod
    def np_array(func):
        @wraps(func)
        def wrapper(y_test, y_pred, *args, **kwargs):
            return func(np.array(y_test), np.array(y_pred), *args, **kwargs)

        return wrapper


class Metrics:

    @staticmethod
    @MetricsWrapper.np_array
    def mean_absolute_error(y_test, y_pred):
        """
        n = len(y_true)
        return sum(abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred)) / n
        """
        return np.mean(np.abs(y_test - y_pred))

    @staticmethod
    @MetricsWrapper.np_array
    def mean_squared_error(y_test, y_pred):
        """
        n = len(y_true)
        return sum((y_t - y_p)**2 for y_t, y_p in zip(y_true, y_pred)) / n
        """
        return np.mean((y_test - y_pred) ** 2)

    @staticmethod
    @MetricsWrapper.np_array
    def r2_score(y_test, y_pred):
        """
        mean_y_true = sum(y_true) / len(y_true)
        numerator = sum((y_t - y_p)**2 for y_t, y_p in zip(y_true, y_pred))
        denominator = sum((y_t - mean_y_true)**2 for y_t in y_true)
        r_squared = 1 - (numerator / denominator)
        return r_squared
        """
        numerator = np.sum((y_test - y_pred) ** 2)
        denominator = np.sum((y_test - np.mean(y_pred)) ** 2)
        return 1 - (numerator / denominator)

    @staticmethod
    @MetricsWrapper.np_array
    def max_error(y_test, y_pred):
        """
        max_err = max(abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred))
        return max_err
        """
        return np.max(np.abs(y_test - y_pred))

    @staticmethod
    def show_our_metrics(model_name, y_test, y_pred):
        print("\n-------\n")
        print(f"{model_name} MAE: {Metrics.mean_absolute_error(y_test, y_pred)}")
        print(f"{model_name} MSE: {Metrics.mean_squared_error(y_test, y_pred)}")
        print(f"{model_name} R2: {Metrics.r2_score(y_test, y_pred)}")
        print(f"{model_name} Max Error: {Metrics.max_error(y_test, y_pred)}")
        print("\n-------\n")
