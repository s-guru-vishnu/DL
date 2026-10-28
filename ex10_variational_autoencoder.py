import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

tf.keras.utils.set_random_seed(1)


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))

        return (
            z_mean
            + tf.exp(0.5 * z_log_var) * epsilon
        )


# Encoder
encoder_input = keras.Input(shape=(8, 8, 1))

x = layers.Flatten()(encoder_input)
x = layers.Dense(32, activation="relu")(x)

z_mean = layers.Dense(2, name="z_mean")(x)
z_log_var = layers.Dense(2, name="z_log_var")(x)

z = Sampling()([z_mean, z_log_var])

encoder = keras.Model(
    encoder_input,
    [z_mean, z_log_var, z],
    name="encoder"
)


# Decoder
decoder_input = keras.Input(shape=(2,))

x = layers.Dense(32, activation="relu")(decoder_input)
x = layers.Dense(64, activation="sigmoid")(x)
decoder_output = layers.Reshape((8, 8, 1))(x)

decoder = keras.Model(
    decoder_input,
    decoder_output,
    name="decoder"
)


class VariationalAutoencoder(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def call(self, inputs, training=False):
        _, _, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)

    def calculate_losses(self, x, training=False):
        mean, log_var, z = self.encoder(x, training=training)
        reconstruction = self.decoder(z, training=training)

        bce = keras.losses.binary_crossentropy(x, reconstruction)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(bce, axis=[1, 2])
        )

        kl_per_sample = -0.5 * tf.reduce_sum(
            1
            + log_var
            - tf.square(mean)
            - tf.exp(log_var),
            axis=1
        )

        kl_loss = tf.reduce_mean(kl_per_sample)

        total_loss = reconstruction_loss + kl_loss

        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data

        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = (
                self.calculate_losses(x, training=True)
            )

        gradients = tape.gradient(
            total_loss,
            self.trainable_variables
        )

        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(
            reconstruction_loss
        )
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            metric.name: metric.result()
            for metric in self.metrics
        }

    def test_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data

        total_loss, reconstruction_loss, kl_loss = (
            self.calculate_losses(x, training=False)
        )

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(
            reconstruction_loss
        )
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            metric.name: metric.result()
            for metric in self.metrics
        }


# Prepare images
digits = load_digits()

X = digits.images[..., np.newaxis].astype(np.float32) / 16.0

X_train, X_test = train_test_split(
    X,
    test_size=0.1,
    random_state=1
)

vae = VariationalAutoencoder(encoder, decoder)

vae.compile(
    optimizer=keras.optimizers.Adam()
)

history = vae.fit(
    X_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_test, X_test),
    verbose=2
)

test_results = vae.evaluate(
    X_test,
    X_test,
    verbose=0,
    return_dict=True
)

print(
    "Final Training Loss:",
    round(float(history.history["loss"][-1]), 4)
)

print(
    "Reconstruction Loss:",
    round(float(test_results["reconstruction_loss"]), 4)
)

