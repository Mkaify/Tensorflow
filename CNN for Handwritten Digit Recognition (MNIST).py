import tensorflow as tf
import matplotlib.pyplot as plt


# ================================
# Load MNIST Dataset
# ================================
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


# ================================
# Normalize & Reshape
# ================================
X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train[..., tf.newaxis]   # (28,28,1)
X_test = X_test[..., tf.newaxis]


# ================================
# Build CNN Model
# ================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),
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
# Train Model
# ================================
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_split=0.1,
    verbose=1
)


# ================================
# Evaluate
# ================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.3f}")


# ================================
# Visualize Predictions
# ================================
preds = model.predict(X_test[:5])

for i, pred in enumerate(preds):

    plt.imshow(X_test[i].squeeze(), cmap="gray")

    plt.title(
        f"Predicted: {tf.argmax(pred).numpy()}, "
        f"True: {y_test[i]}"
    )

    plt.axis("off")
    plt.show()
