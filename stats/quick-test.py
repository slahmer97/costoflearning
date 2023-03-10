import pandas as pd
import numpy as np


def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))


x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

print(X)
print(Y)
print(Z)
