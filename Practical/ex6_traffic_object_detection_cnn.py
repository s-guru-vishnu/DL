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