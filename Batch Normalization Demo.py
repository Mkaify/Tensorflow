import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Reproducibility
# ----------------------------
np.random.seed(42)
tf.random.set_seed(42)

# ----------------------------
# Generate synthetic data
# ----------------------------
X = np.linspace(-3, 3, 500).reshape(-1, 1).astype(np.float32)
y = (X[:, 0] > 0).astype(np.float32).reshape(-1, 1)

# Add small Gaussian noise
y += np.random.normal(0, 0.05, size=y.shape)

# ----------------------------
# Model WITHOUT Batch Normalization
# ----------------------------
model_no_bn = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# ----------------------------
# Model WITH Batch Normalization
# ----------------------------
model_with_bn = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(32),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# ----------------------------
# Compile models
# ----------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model_no_bn.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"],
    run_eagerly=True
)

model_with_bn.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"],
    run_eagerly=True
)

# ----------------------------
# Train models
# ----------------------------
history_no_bn = model_no_bn.fit(
    X, y,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

history_with_bn = model_with_bn.fit(
    X, y,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# ----------------------------
# Plot comparison
# ----------------------------
plt.figure(figsize=(8, 5))

plt.plot(history_no_bn.history["val_accuracy"], label="Without Batch Normalization")
plt.plot(history_with_bn.history["val_accuracy"], label="With Batch Normalization")

plt.title("Effect of Batch Normalization on Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)

plt.show()
