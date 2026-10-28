import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, Dense
from tensorflow.keras.models import Sequential
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

tf.keras.utils.set_random_seed(1)

xr, xe, yr, ye = train_test_split(
    *load_breast_cancer(return_X_y=True),
    random_state=1
)

m = Sequential([
    Input(shape=(30,)),
    BatchNormalization(),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

m.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

m.fit(
    xr, yr,
    epochs=30,
    batch_size=16,
    validation_split=0.1,
    verbose=0
)

loss, accuracy, precision, recall = m.evaluate(xe, ye, verbose=0)

print("Test Loss:", round(float(loss), 4))
print("Test Accuracy:", round(float(accuracy), 4))
print("Test Precision:", round(float(precision), 4))
print("Test Recall:", round(float(recall), 4))

