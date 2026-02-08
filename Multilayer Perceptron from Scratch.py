import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Reproducibility
tf.keras.utils.set_random_seed(42)

# Load Iris dataset
iris = load_iris()
X = iris.data                      # Shape: (150, 4)
y = iris.target                    # Shape: (150,)

# One-hot encode labels (modern Keras way)
y_encoded = tf.keras.utils.to_categorical(y, num_classes=3)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Define the MLP model (modern style)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    verbose=0
)

# Evaluate on test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.2f}")
