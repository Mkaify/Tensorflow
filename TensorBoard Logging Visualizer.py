import tensorflow as tf
import datetime
import os
import numpy as np


# ================================
# Set Random Seeds (Reproducibility)
# ================================
np.random.seed(42)
tf.random.set_seed(42)


# ================================
# Load & Preprocess MNIST
# ================================
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Add channel dimension
X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]


# ================================
# Build CNN Model
# ================================
model = tf.keras.Sequential([

    # Convolution Block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),

    # Convolution Block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    # Classifier
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])


# ================================
# Compile Model
# ================================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ================================
# TensorBoard Setup
# ================================
log_dir = os.path.join(
    "logs",
    "tensorboard_demo",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True
)


# ================================
# Additional Callbacks
# ================================
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=3,
    restore_best_weights=True
)


# ================================
# Train Model
# ================================
history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.2,
    callbacks=[tensorboard_cb, early_stop],
    verbose=1
)


# ================================
# Evaluate on Test Data
# ================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Accuracy: {test_acc:.4f}")


# ================================
# TensorBoard Command
# ================================
print("\nRun this in terminal to open TensorBoard:")
print("tensorboard --logdir=logs/tensorboard_demo/")
