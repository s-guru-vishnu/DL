# Deep Learning Laboratory (Verified Compact 10-Line Implementations)

All 10 deep learning lab experiments tailored for academic laboratory records:
- **Strictly within 10 lines of code** per experiment
- **Zero semicolons (`;`)**
- **100% verified clean execution** with actual metrics and zero runtime errors

---

## Experiment Summary & Code

### Ex 1: Supervised Classification Using Logistic Regression
**File:** [`ex1_supervised_classification.py`](ex1_supervised_classification.py) (10 lines)
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
X, x, Y, y = train_test_split(*load_breast_cancer(return_X_y=True), random_state=1)
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, Y)
print("Score:", m.score(x, y))
print("Confusion Matrix:\n", confusion_matrix(y, m.predict(x)))
```

### Ex 2: Deep Learning Model Using Feed-Forward Neural Network
**File:** [`ex2_deep_learning_model.py`](ex2_deep_learning_model.py) (9 lines)
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from keras import Sequential, layers
X, x, Y, y = train_test_split(*load_breast_cancer(return_X_y=True), random_state=1)
m = Sequential([layers.Input((30,)), layers.BatchNormalization(), layers.Dense(16, activation='relu'), layers.Dense(8, activation='relu'), layers.Dense(1, activation='sigmoid')])
m.compile('adam', 'binary_crossentropy', metrics=['accuracy', 'precision', 'recall'])
m.fit(X, Y, epochs=30, batch_size=16, verbose=0)
res = dict(zip(['Loss', 'Accuracy', 'Precision', 'Recall'], m.evaluate(x, y, verbose=0)))
print("\n".join(f"Test {k}: {v:.4f}" for k, v in res.items()))
```

### Ex 3: Forward Propagation Using NumPy
**File:** [`ex3_forward_propagation_numpy.py`](ex3_forward_propagation_numpy.py) (10 lines)
```python
import numpy as np
from scipy.special import expit
np.random.seed(1)
X = np.array([[0.5, 0.1, 0.9], [0.2, 0.8, 0.4]])
W1, W2 = np.random.randn(3, 4) * 0.5, np.random.randn(4, 1) * 0.5
A1 = np.maximum(0, X @ W1)
A2 = expit(A1 @ W2)
print("Hidden Layer Output:\n", A1.round(4))
print("Final Output:\n", A2.round(4))
print("Predicted Classes:", (A2 > 0.5).astype(int).ravel())
```

### Ex 4: Image Classification Using CNN
**File:** [`ex4_image_classification_cnn.py`](ex4_image_classification_cnn.py) (10 lines)
```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras import Sequential, layers
d = load_digits()
X, x, Y, y = train_test_split(d.images[..., None] / 16.0, d.target, test_size=0.2, random_state=1, stratify=d.target)
m = Sequential([layers.Input((8, 8, 1)), layers.Conv2D(16, 3, activation='relu'), layers.MaxPooling2D(), layers.Flatten(), layers.Dense(10, 'softmax')])
m.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
m.fit(X, Y, epochs=10, batch_size=32, verbose=0)
loss, acc = m.evaluate(x, y, verbose=0)
print(f"Test Loss: {loss:.4f}\nTest Accuracy: {acc:.4f}")
```

### Ex 5: Text Classification Using CNN
**File:** [`ex5_text_classification_cnn.py`](ex5_text_classification_cnn.py) (10 lines)
```python
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
```

### Ex 6: Traffic Object Detection CNN & NMS
**File:** [`ex6_traffic_object_detection_cnn.py`](ex6_traffic_object_detection_cnn.py) (10 lines)
```python
import numpy as np, tensorflow as tf
from keras import layers, models
c = np.random.randint(0, 2, 200)
X = np.array([np.pad(np.ones((5 if k==0 else 7, 8 if k==0 else 14, 1)), ((10, 17 if k==0 else 15), (10, 14 if k==0 else 8), (0, 0))) for k in c], dtype='float32')
b = np.where(c[:, None] == 0, [0.31, 0.31, 0.56, 0.47], [0.25, 0.19, 0.75, 0.53]).astype('float32')
feat = layers.Flatten()(layers.Conv2D(8, 3, activation='relu')(inp := layers.Input((32, 32, 1))))
m = models.Model(inp, [layers.Dense(2, 'softmax')(feat), layers.Dense(4, 'sigmoid')(feat)])
m.compile('adam', loss=['categorical_crossentropy', 'mse'], metrics=[['accuracy'], []])
m.fit(X, [tf.keras.utils.to_categorical(c, 2), b], epochs=10, verbose=0)
print("Accuracy:", round(m.evaluate(X, [tf.keras.utils.to_categorical(c, 2), b], verbose=0)[3], 4), "\nSelected NMS Boxes:", tf.image.non_max_suppression(tf.constant([[0.,0.,9.,9.],[1.,1.,10.,10.],[20.,20.,30.,30.]]), [0.9, 0.8, 0.7], 3).numpy())
```

### Ex 7: Sequence Modeling Using LSTM
**File:** [`ex7_rnn_lstm.py`](ex7_rnn_lstm.py) (10 lines)
```python
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
```

### Ex 8: Image Augmentation Using RBM
**File:** [`ex8_rbm_image_augmentation.py`](ex8_rbm_image_augmentation.py) (10 lines)
```python
import numpy as np, scipy.special as sp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
(X, x, Y, y), rbm = train_test_split(load_digits().data / 16.0, load_digits().target, train_size=300, random_state=0), BernoulliRBM(64, n_iter=10, random_state=1)
rbm.fit(X)
Xg = sp.expit(rbm.gibbs(rbm.transform(X)) @ rbm.components_ + rbm.intercept_visible_)
b = LogisticRegression(max_iter=1000).fit(X, Y).score(x, y)
print("Without RBM:", round(b, 4), "With RBM:", round(LogisticRegression(max_iter=1000).fit(np.vstack([X, Xg]), np.concatenate([Y, Y])).score(x, y), 4))
```

### Ex 9: Sentiment Analysis Using LSTM
**File:** [`ex9_sentiment_lstm.py`](ex9_sentiment_lstm.py) (10 lines)
```python
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
```

### Ex 10: Variational Autoencoder (VAE)
**File:** [`ex10_variational_autoencoder.py`](ex10_variational_autoencoder.py) (10 lines)
```python
import keras, tensorflow as tf
from sklearn.datasets import load_digits
X = load_digits().data / 16.0
inp = keras.Input((64,))
zm, zv = keras.layers.Dense(2)(inp), keras.layers.Dense(2)(inp)
z = keras.layers.Lambda(lambda a: a[0] + tf.exp(0.5 * a[1]) * tf.random.normal(tf.shape(a[0])))([zm, zv])
out = keras.layers.Dense(64, 'sigmoid')(keras.layers.Dense(32, 'relu')(z))
m = keras.Model(inp, keras.layers.Concatenate()([out, zm, zv]))
m.compile('adam', lambda y, p: tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(y, p[:,:64]), -1) - 0.5 * tf.reduce_sum(1 + p[:,66:] - tf.square(p[:,64:66]) - tf.exp(p[:,66:]), -1)))
print("Final VAE Loss:", round(m.fit(X, X, epochs=10, batch_size=64, verbose=0).history['loss'][-1], 4))
```
