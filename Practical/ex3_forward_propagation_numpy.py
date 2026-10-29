import numpy as np
from scipy.special import expit
np.random.seed(1)
X = np.array([[0.5, 0.1, 0.9], [0.2, 0.8, 0.4]])
W1, W2 = np.random.randn(3, 4) * 0.5, np.random.randn(4, 1) * 0.5
A1 = np.maximum(0, X @ W1)
A2 = expit(A1 @ W2)
print("Hidden Layer Output:\n", A1.round(4))
print("Final Output:\n", A2.round(4))
print("Predicted Classes:", (A2 > 0.5).astype(int).ravel())