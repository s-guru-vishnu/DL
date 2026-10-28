import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

tf.keras.utils.set_random_seed(1)

d = load_digits()

X = d.images[..., None].astype("float32") / 16.0
y = d.target

xr, xe, yr, ye = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1,
    stratify=y
)

m = Sequential([
    Input(shape=(8, 8, 1)),
    Conv2D(16, kernel_size=3, activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(10, activation="softmax")
])

m.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

m.fit(
    xr, yr,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)

loss, accuracy = m.evaluate(xe, ye, verbose=0)

print("Test Loss:", round(float(loss), 4))
print("Test Accuracy:", round(float(accuracy), 4))

