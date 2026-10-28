import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, TextVectorization
)
from tensorflow.keras.models import Sequential

tf.keras.utils.set_random_seed(1)

positive = [
    "great movie",
    "fine acting",
    "great acting",
    "excellent film",
    "wonderful story",
    "good movie",
    "amazing acting",
    "nice film"
]

negative = [
    "awful movie",
    "poor acting",
    "awful acting",
    "terrible film",
    "bad story",
    "horrible movie",
    "weak acting",
    "boring film"
]

base_texts = positive + negative
base_labels = [1] * len(positive) + [0] * len(negative)

texts = base_texts * 20
labels = np.array(base_labels * 20, dtype=np.float32)

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
    LSTM(8),
    Dense(1, activation="sigmoid")
])

m.compile(
    optimizer="adam",
    loss="binary_crossentropy",
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
    "great acting",
    "awful acting"
])

probabilities = m.predict(
    vectorizer(test_texts),
    verbose=0
).ravel()

for sentence, probability in zip(test_texts, probabilities):
    sentiment = "Positive" if probability >= 0.5 else "Negative"

    print("Sentence:", sentence)
    print("Positive Sentiment Probability:", round(float(probability), 4))
    print("Predicted Sentiment:", sentiment)
    print()

