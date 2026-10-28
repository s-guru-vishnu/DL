import numpy as np, keras
from keras.layers import Input, Embedding, LSTM, Dense, TextVectorization
texts, labels = ["great movie", "fine acting", "excellent film", "wonderful story", "awful movie", "poor acting", "terrible film", "boring film"] * 20, np.array([1, 1, 1, 1, 0, 0, 0, 0] * 20)
v = TextVectorization(max_tokens=100, output_sequence_length=2)
v.adapt(texts)
m = keras.Sequential([Input((2,)), Embedding(len(v.get_vocabulary()), 8), LSTM(8), Dense(1, 'sigmoid')])
m.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
m.fit(v(texts), labels, epochs=15, batch_size=16, verbose=0)
tests = ["great acting", "awful acting"]
print("\n".join(f"{s}: {'Positive' if p>=0.5 else 'Negative'} ({p:.4f})" for s, p in zip(tests, m.predict(v(tests), verbose=0).ravel())))