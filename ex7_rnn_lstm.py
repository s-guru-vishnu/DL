import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Sequential

tf.keras.utils.set_random_seed(1)

text = "deep learning is fun. " * 20

chars = np.array(sorted(set(text)))
char_to_id = {char: i for i, char in enumerate(chars)}

ids = np.array(
    [char_to_id[char] for char in text],
    dtype=np.int32
)

sequence_length = 10

X_ids = np.array([
    ids[i:i + sequence_length]
    for i in range(len(ids) - sequence_length)
])

y = ids[sequence_length:]

X = np.eye(len(chars), dtype=np.float32)[X_ids]

m = Sequential([
    Input(shape=(sequence_length, len(chars))),
    LSTM(64),
    Dense(len(chars), activation="softmax")
])

m.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

m.fit(
    X, y,
    epochs=20,
    batch_size=32,
    verbose=0
)

predictions = m.predict(X[:30], verbose=0)
predicted_ids = predictions.argmax(axis=1)

print("Predicted Text:")
print("".join(chars[predicted_ids]))

print("Actual Text:")
print("".join(chars[y[:30]]))

