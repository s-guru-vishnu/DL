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