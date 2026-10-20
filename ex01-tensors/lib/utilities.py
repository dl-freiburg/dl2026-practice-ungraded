import numpy as np
from typing import Tuple


# true weights of the data generators below (lib/plot.py shows them for comparison)
W_TRUE_1D = np.array([1.5])
W_TRUE_2D = np.array([1.5, 0.5])


# data generation
def data_1d() -> Tuple[np.array, np.array]:
    # generates y data according to the example in Figure 5.1.

    # the data generator; Input: np.array of shape (1)
    def generator(x):
        # data checking
        if isinstance(x, list):
            x = np.array(x)
        if len(x.shape) != 1 or len(x) != 1:
            raise ValueError('x should be a vector of size 1')
        return W_TRUE_1D[0] * x[0] + np.random.normal(0.0, 0.1)

    # the data
    X_train = np.array([[-0.4], [-0.35], [-0.3], [-0.3],
                        [-0.2], [0.1], [0.2], [0.3], [0.4], [0.6]])
    y_train = np.array([generator(X_train[i]) for i in range(len(X_train))])
    return X_train, y_train


def data_2d(noise_rate: float = 0.1) -> Tuple[np.array, np.array]:
    # generates random data according to the function y = 1.5 * x1 + 0.5 * x2
    # + epsilon

    def generator(x):
        # data checking
        if isinstance(x, list):
            x = np.array(x)
        if len(x.shape) != 1 or len(x) != 2:
            raise ValueError('x should be a vector of size 2')
        return W_TRUE_2D[0] * x[0] + W_TRUE_2D[1] * x[1] + np.random.normal(0.0, noise_rate)

    data_n = 40
    X_train = np.reshape(np.random.random(data_n * 2), (data_n, 2))
    y_train = np.array([generator(X_train[i]) for i in range(len(X_train))])
    return X_train, y_train
