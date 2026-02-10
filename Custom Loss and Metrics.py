import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic regression data
X = np.linspace(1, 10, 100, dtype=np.float32)
y = 5 * X + 10 + np.random.randn(*X.shape).astype(np.float32) * 2

# Custom MAPE loss (numerically stable)
def custom_mape(y_true, y_pred):
    epsilon = tf.keras.backend.epsilon()
    return tf.reduce_mean(tf.abs((y_true - y_pred) / tf.maximum(tf.abs(y_true), epsilon))) * 100

# Custom R² metric
def r2_score(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - ss_res / ss_tot

# Model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.float32),
    tf.keras.layers.Dense(1)
])

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
    loss=custom_mape,
    metrics=[r2_score]
)

# Train
model.fit(X, y, epochs=100, verbose=0)

# Predict
preds = model.predict(X)

# Plot
plt.scatter(X, y, label="True Data")
plt.plot(X, preds, label="Predicted Line")
plt.title("Custom MAPE Loss & R² Metric")
plt.legend()
plt.show()
