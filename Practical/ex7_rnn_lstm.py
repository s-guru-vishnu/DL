import numpy as np, keras, tensorflow as tf
tf.random.set_seed(1)
text, chars = "deep learning is fun. " * 10, sorted(set("deep learning is fun. "))
ids = np.array([chars.index(c) for c in text])
X, y = np.eye(len(chars))[[ids[i:i+10] for i in range(len(ids)-10)]], ids[10:]
m = keras.Sequential([keras.layers.Input((10, len(chars))), keras.layers.LSTM(64), keras.layers.Dense(len(chars), 'softmax')])
m.compile('adam', 'sparse_categorical_crossentropy')
m.fit(X, y, epochs=35, verbose=0)
pred = np.array(chars)[m.predict(X[:25], verbose=0).argmax(1)]
print("Predicted:", "".join(pred), "\nActual:   ", "".join(np.array(chars)[y[:25]]))