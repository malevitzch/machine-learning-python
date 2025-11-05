from math import sin, pi
import numpy as np
import torch as tc
import matplotlib.pyplot as plt


def sin_poly(a, b):
    def fn(x):
        return a * x + b * (x**3)

    return fn


def fit(X, Y, iters=1000, lr=0.10):
    a = tc.tensor(1, dtype=tc.float64, requires_grad=True)
    b = tc.tensor(0, dtype=tc.float64, requires_grad=True)
    optimizer = tc.optim.SGD([a, b], lr=lr)

    for i in range(iters):
        Y_pred = a * X + b * (X**3)
        loss = tc.mean((Y - Y_pred) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return a.detach().numpy(), b.detach().numpy()


X = np.array([-pi / 2 + i * pi / 20 for i in range(21)])
Y = np.vectorize(sin)(X)
print(Y)

X_t = tc.tensor(X, dtype=tc.float64)
Y_t = tc.tensor(Y, dtype=tc.float64)

a, b = fit(X_t, Y_t)
print(f"{a}*x + {b}*x^3")

fn = sin_poly(a, b)
xs = np.linspace(-pi / 2, pi / 2, 100)
ys = fn(xs)
plt.plot(xs, ys)

real_ys = np.vectorize(sin)(xs)
plt.plot(xs, real_ys)

plt.show()
