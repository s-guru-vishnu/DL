import numpy as np, keras
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, TextVectorization
texts, labels = ["win goal", "team score", "new chip", "digital program", "eat pills", "take medicine"] * 20, np.array([0, 0, 1, 1, 2, 2] * 20)
v = TextVectorization(max_tokens=100, output_sequence_length=2)
v.adapt(texts)
m = keras.Sequential([Input((2,)), Embedding(len(v.get_vocabulary()), 8), Conv1D(16, 2, activation='relu'), GlobalMaxPooling1D(), Dense(3, 'softmax')])
m.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
m.fit(v(texts), labels, epochs=20, batch_size=16, verbose=0)
tests = np.array(["win goal", "new chip", "eat pills"])
print(dict(zip(tests, np.array(["Sports", "Technology", "Health"])[m.predict(v(tests), verbose=0).argmax(1)])))