import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ================================
# Set Random Seeds
# ================================
np.random.seed(42)
tf.random.set_seed(42)


# ================================
# Load Dataset
# ================================
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


# ================================
# Normalize Data
# ================================
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# ================================
# Build Model
# ================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),          # Proper Input Layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),            # Regularization
    tf.keras.layers.Dense(10, activation='softmax')
])


# ================================
# Compile Model
# ================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ================================
# Callbacks (Early Stopping)
# ================================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


# ================================
# Train Model
# ================================
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)


# ================================
# Evaluate
# ================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Accuracy: {test_acc:.3f}")


# ================================
# Display Sample Predictions
# ================================
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

preds = model.predict(X_test[:5], verbose=0)

plt.figure(figsize=(10, 3))

for i in range(5):

    plt.subplot(1, 5, i + 1)

    plt.imshow(X_test[i], cmap="gray")
    plt.title(
        f"P: {class_names[preds[i].argmax()]}\nT: {class_names[y_test[i]]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()
