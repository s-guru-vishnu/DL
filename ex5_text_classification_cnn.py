import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization

tf.keras.utils.set_random_seed(1)

sports = [
    "win goal", "score match", "fast player",
    "team victory", "football match", "player goal",
    "win game", "team score"
]

technology = [
    "new chip", "new app", "computer program",
    "software update", "mobile device", "digital technology",
    "computer chip", "smart application"
]

health = [
    "eat pills", "take pills", "healthy food",
    "fresh fruits", "medical treatment", "health care",
    "take medicine", "eat fruits"
]

base_texts = sports + technology + health
base_labels = (
    [0] * len(sports)
    + [1] * len(technology)
    + [2] * len(health)
)

texts = base_texts * 15
labels = np.array(base_labels * 15, dtype=np.int32)

vectorizer = TextVectorization(
    max_tokens=100,
    output_mode="int",
    output_sequence_length=2
)

vectorizer.adapt(np.array(texts))

X = vectorizer(np.array(texts))
vocab_size = len(vectorizer.get_vocabulary())

m = Sequential([
    Input(shape=(2,), dtype="int64"),
    Embedding(input_dim=vocab_size, output_dim=8),
    Conv1D(16, kernel_size=2, activation="relu"),
    GlobalMaxPooling1D(),
    Dense(3, activation="softmax")
])

m.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

m.fit(
    X, labels,
    epochs=15,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

test_texts = np.array([
    "win goal",
    "new chip",
    "eat pills"
])

predictions = m.predict(vectorizer(test_texts), verbose=0)
classes = predictions.argmax(axis=1)

categories = {
    0: "Sports",
    1: "Technology",
    2: "Health"
}

for sentence, category_id in zip(test_texts, classes):
    print("Sentence:", sentence)
    print("Predicted Category:", categories[int(category_id)])
    print()

