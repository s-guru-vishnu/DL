from sklearn.datasets import *
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.pipeline import *
from sklearn.metrics import *

xr, xe, yr, ye = train_test_split(
    *load_breast_cancer(return_X_y=True),
    random_state=1
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
).fit(xr, yr)

print(model.score(xe, ye))
print(confusion_matrix(ye, model.predict(xe)))

