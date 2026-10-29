import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, Dense
from sklearn.neural_network import BernoulliRBM
rnn = Sequential([Input(shape=(10, 4)), SimpleRNN(8)])
lstm = Sequential([Input(shape=(10, 4)), LSTM(8), Dense(1)])
rbm = BernoulliRBM(n_components=8, n_iter=10)
rbm.fit(np.random.rand(100, 20))
x = Input(shape=(20,))
z = Dense(8, activation="relu")(x)
mean = Dense(2)(z)
log_var = Dense(2)(z)
sample = mean + tf.random.normal(tf.shape(mean)) * tf.exp(log_var / 2)
decoded = Dense(20, activation="sigmoid")(sample)
vae = Model(x, decoded)
vae.compile(optimizer="adam", loss="binary_crossentropy")