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