from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from keras import Sequential, layers
X, x, Y, y = train_test_split(*load_breast_cancer(return_X_y=True), random_state=1)
m = Sequential([layers.Input((30,)), layers.BatchNormalization(), layers.Dense(16, activation='relu'), layers.Dense(8, activation='relu'), layers.Dense(1, activation='sigmoid')])
m.compile('adam', 'binary_crossentropy', metrics=['accuracy', 'precision', 'recall'])
m.fit(X, Y, epochs=30, batch_size=16, verbose=0)
res = dict(zip(['Loss', 'Accuracy', 'Precision', 'Recall'], m.evaluate(x, y, verbose=0)))
print("\n".join(f"Test {k}: {v:.4f}" for k, v in res.items()))