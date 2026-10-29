import numpy as np, scipy.special as sp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
(X, x, Y, y), rbm = train_test_split(load_digits().data / 16.0, load_digits().target, train_size=300, random_state=0), BernoulliRBM(64, n_iter=10, random_state=1)
rbm.fit(X)
Xg = sp.expit(rbm.gibbs(rbm.transform(X)) @ rbm.components_ + rbm.intercept_visible_)
b = LogisticRegression(max_iter=1000).fit(X, Y).score(x, y)
print("Without RBM:", round(b, 4), "With RBM:", round(LogisticRegression(max_iter=1000).fit(np.vstack([X, Xg]), np.concatenate([Y, Y])).score(x, y), 4))