import torch as t


def linear_function(a, b):
    def f(x):
        return a*x + b
    return f


print(linear_function(10, 15)(3))
