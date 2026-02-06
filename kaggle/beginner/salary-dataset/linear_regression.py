import numpy as np


def manual_linear_regression(x, y):
    x = np.array(x)
    y = np.array(y)

    x_mean = x.mean()
    y_mean = y.mean()

    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()

    b1 = numerator / denominator
    b0 = y_mean - b1 * x_mean

    return b0, b1


def predict(x, b0, b1):
    return b0 + b1 * x
