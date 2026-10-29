import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Conv1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, GlobalMaxPooling1D
image_model = Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    Conv2D(16, 3, activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(10, activation="softmax")
])
text_model = Sequential([
    Embedding(1000, 16, input_length=20),
    Conv1D(16, 3, activation="relu"),
    GlobalMaxPooling1D(),
    Dense(4, activation="softmax")
])