import torch as t
import numpy as np
import matplotlib.pyplot as plt


def linear_function(a, b):
    def f(x):
        return a * x + b

    return f


def deviated_points_uniform(f, X, percent_error):
    deviation = (percent_error / 100) * np.random.uniform(-1, 1, size=X.shape)
    Y = np.vectorize(f)(X) * (1 + deviation)
    return Y


def fit_line(X, Y, iters=1000, lr=0.01):
    y_scale_factor = np.max(abs(Y))
    Y_scaled = Y / y_scale_factor
    b = t.tensor(Y_scaled[0], requires_grad=True)
    slope = (Y_scaled[-1] - Y_scaled[0]) / (X[-1] - X[0])
    a = t.tensor(slope, requires_grad=True)
    Y_t = t.tensor(Y_scaled, dtype=t.float64)
    X_t = t.tensor(X, dtype=t.float64)
    optimizer = t.optim.SGD([a, b], lr=lr)

    for i in range(iters):
        Y_pred = a * X_t + b
        loss = t.mean((Y_t - Y_pred) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (a.detach().numpy() * y_scale_factor, b.detach().numpy() * y_scale_factor)


def plot(X, Y, a, b):
    plt.figure(figsize=(8, 5))
    plt.scatter(X, Y, label="Data", color="red")
    plt.plot(X, np.vectorize(linear_function(a, b))(X), label="Line fit", color="blue")
    plt.legend()
    plt.show()


a = 1.5
b = 3
X = np.array(range(1, 15))
Y = np.array(deviated_points_uniform(linear_function(a, b), X, 10))

for a, b in zip(X, Y):
    print(a, b, sep=" ", end="\n")

(a, b) = fit_line(X, Y, 1000, 0.01)
print(a)
print(b)
plot(X, Y, a, b)
