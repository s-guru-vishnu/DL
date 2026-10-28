import numpy as np
from scipy.special import expit
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

d = load_digits()

X = d.data.astype(np.float64) / 16.0
y = d.target

Xs, Xq, ys, yq = train_test_split(
    X,
    y,
    train_size=300,
    random_state=0,
    stratify=y
)

rbm1 = BernoulliRBM(
    n_components=100,
    learning_rate=0.05,
    n_iter=20,
    random_state=1,
    verbose=False
)

rbm1.fit(Xs)

hidden = rbm1.transform(Xs)

rbm2 = BernoulliRBM(
    n_components=64,
    learning_rate=0.05,
    n_iter=20,
    random_state=2,
    verbose=False
)

rbm2.fit(hidden)

# Approximate synthetic samples obtained through the RBM layers.
# Synthetic samples are approximate and may not improve accuracy.
hidden_samples = rbm2.gibbs(hidden)

Xg = expit(
    hidden_samples @ rbm1.components_
    + rbm1.intercept_visible_
)

baseline = LogisticRegression(
    max_iter=2000,
    random_state=1
)

baseline.fit(Xs, ys)
baseline_predictions = baseline.predict(Xq)

augmented = LogisticRegression(
    max_iter=2000,
    random_state=1
)

augmented.fit(
    np.vstack([Xs, Xg]),
    np.concatenate([ys, ys])
)

augmented_predictions = augmented.predict(Xq)

print(
    "Accuracy Without RBM Augmentation:",
    round(accuracy_score(yq, baseline_predictions), 4)
)

print(
    "Accuracy With RBM Augmentation:",
    round(accuracy_score(yq, augmented_predictions), 4)
)

