from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
X, x, Y, y = train_test_split(*load_breast_cancer(return_X_y=True), random_state=1)
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, Y)
print("Score:", m.score(x, y))
print("Confusion Matrix:\n", confusion_matrix(y, m.predict(x)))