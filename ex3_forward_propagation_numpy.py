import numpy as np

np.random.seed(1)

X = np.array([
    [0.5, 0.1, 0.9],
    [0.2, 0.8, 0.4]
])

W1 = np.random.randn(3, 4) * 0.5
b1 = np.zeros(4)

W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros(1)

A1 = np.maximum(0, X @ W1 + b1)

Z2 = A1 @ W2 + b2
A2 = 1 / (1 + np.exp(-Z2))

print("Hidden Layer Output:")
print(A1.round(4))

print("Final Output:")
print(A2.round(4))

print("Predicted Classes:")
print((A2 > 0.5).astype(int).ravel())

