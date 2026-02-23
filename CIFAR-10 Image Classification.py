import tensorflow as tf
import matplotlib.pyplot as plt


# ================================
# Load CIFAR-10 Dataset
# ================================
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()


# ================================
# Normalize & Reshape Labels
# ================================
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = y_train.squeeze()
y_test = y_test.squeeze()


# ================================
# Class Names
# ================================
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ================================
# Build CNN Model
# ================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),              # Reduce overfitting

    tf.keras.layers.Dense(10, activation="softmax")
])


# ================================
# Compile
# ================================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ================================
# Train
# ================================
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_split=0.1,
    verbose=1
)


# ================================
# Evaluate
# ================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Accuracy: {test_acc:.3f}")


# ================================
# Predict & Visualize
# ================================
preds = model.predict(X_test[:5])

for i, pred in enumerate(preds):

    predicted_label = class_names[tf.argmax(pred).numpy()]
    true_label = class_names[y_test[i]]

    plt.imshow(X_test[i])

    plt.title(f"Predicted: {predicted_label}\nTrue: {true_label}")

    plt.axis("off")
    plt.show()
