import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

tf.keras.utils.set_random_seed(1)

rng = np.random.default_rng(1)

n = 2000
images = np.zeros((n, 32, 32, 1), dtype=np.float32)

class_ids = rng.integers(0, 2, size=n)

boxes = np.zeros((n, 4), dtype=np.float32)

for i, class_id in enumerate(class_ids):

    if class_id == 0:
        width, height = 8, 5       # Car
    else:
        width, height = 14, 7      # Bus

    x1 = int(rng.integers(0, 32 - width + 1))
    y1 = int(rng.integers(0, 32 - height + 1))

    x2 = x1 + width
    y2 = y1 + height

    images[i, y1:y2, x1:x2, 0] = 1.0

    boxes[i] = [
        x1 / 32,
        y1 / 32,
        x2 / 32,
        y2 / 32
    ]

class_targets = tf.keras.utils.to_categorical(class_ids, num_classes=2)

(
    xr, xe,
    yr_class, ye_class,
    yr_box, ye_box
) = train_test_split(
    images,
    class_targets,
    boxes,
    test_size=0.2,
    random_state=1,
    stratify=class_ids
)

inputs = layers.Input(shape=(32, 32, 1))

x = layers.Conv2D(8, 3, activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(32, activation="relu")(x)

class_output = layers.Dense(
    2, activation="softmax", name="class_output"
)(x)

box_output = layers.Dense(
    4, activation="sigmoid", name="box_output"
)(x)

m = models.Model(
    inputs=inputs,
    outputs=[class_output, box_output]
)

m.compile(
    optimizer="adam",
    loss={
        "class_output": "categorical_crossentropy",
        "box_output": "mse"
    },
    loss_weights={
        "class_output": 1.0,
        "box_output": 5.0
    },
    metrics={
        "class_output": ["accuracy"],
        "box_output": ["mae"]
    }
)

m.fit(
    xr,
    {
        "class_output": yr_class,
        "box_output": yr_box
    },
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)

class_predictions, box_predictions = m.predict(xe, verbose=0)

true_classes = ye_class.argmax(axis=1)
predicted_classes = class_predictions.argmax(axis=1)

accuracy = np.mean(true_classes == predicted_classes)

print("Traffic Classification Accuracy:", round(float(accuracy), 4))
print("First 5 Predicted Bounding Boxes:")
print(np.round(box_predictions[:5], 3))

# Demonstrate Non-Maximum Suppression (NMS).
# TensorFlow boxes use the format [y1, x1, y2, x2].
nms_boxes = tf.constant([
    [0.0, 0.0, 9.0, 9.0],
    [1.0, 1.0, 10.0, 10.0],
    [20.0, 20.0, 30.0, 30.0]
])

scores = tf.constant([0.9, 0.8, 0.7])

selected_indices = tf.image.non_max_suppression(
    boxes=nms_boxes,
    scores=scores,
    max_output_size=3,
    iou_threshold=0.5
)

print("Selected Boxes After NMS:")
print(selected_indices.numpy())

