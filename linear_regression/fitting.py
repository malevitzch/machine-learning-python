import torch as t
import random as rng
import numpy as np
import matplotlib.pyplot as plt


def linear_function(a, b):
    def f(x):
        return a*x + b
    return f


def deviated_points_uniform(f, X, percent_error):
    deviation = (percent_error/100) * np.random.uniform(-1, 1, size=X.shape)
    Y = np.vectorize(f)(X) * (1 + deviation)
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


def plot(X, Y, a, b):
    plt.figure(figsize=(8, 5))
    plt.scatter(X, Y, label='Data', color='red')
    # plt.plot()
    plt.plot(X, np.vectorize(linear_function(a, b))(X),
             label='fit', color='blue')
    plt.show()


a = 1.5
b = 3
X = np.array(range(1, 14))
Y = np.array(deviated_points_uniform(linear_function(a, b), X, 10))

for a, b in zip(X, Y):
    print(a, b, sep=" ", end="\n")

(a, b) = fit_line(X, Y, 1000, 0.01)
print(a)
print(b)
plot(X, Y, a, b)
