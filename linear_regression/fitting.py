import torch as t
import random as rng
import numpy as np


def linear_function(a, b):
    def f(x):
        return a*x + b
    return f


def deviated_points_uniform(f, X, percent_error):
    Y = [f(x)*(1 + (percent_error/100) * rng.uniform(-1, 1)) for x in X]
    return Y


def fit_line(X, Y, iters=100, lr=0.01):
    a = t.tensor(0.0, requires_grad=True)
    b = t.tensor(0.0, requires_grad=True)
    Y_t = t.tensor(Y)
    X_t = t.tensor(X)
    optimizer = t.optim.SGD([a, b], lr=lr)

    for i in range(iters):
        Y_pred = a * X_t + b
        loss = t.mean((Y_t - Y_pred) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (a.detach().numpy(), b.detach().numpy())


print(deviated_points_uniform(
    linear_function(1, 0),
    [1, 2, 3, 4, 5], 10))
(a, b) = fit_line(np.array([1, 2, 3]), np.array([3, 5, 7]), 100, 0.01)
print(a)
print(b)
