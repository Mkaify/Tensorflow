import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
tf.keras.utils.set_random_seed(42)

# Generate synthetic data
X = np.linspace(0, 10, 100, dtype=np.float32).reshape(-1, 1)
y = 4 * X + 1 + np.random.normal(0, 0.5, size=X.shape).astype(np.float32)

# Trainable variables
W = tf.Variable(tf.random.normal((1, 1)), name="weight")
b = tf.Variable(tf.zeros((1,)), name="bias")

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Training step (compiled)
@tf.function
def train_step(x, y_true):
    with tf.GradientTape() as tape:
        y_pred = tf.matmul(x, W) + b
        loss = tf.reduce_mean(tf.square(y_pred - y_true))
    grads = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(grads, [W, b]))
    return loss

# Training loop
epochs = 200
for epoch in range(epochs):
    loss = train_step(X, y)
    if epoch % 50 == 0:
        tf.print("Epoch", epoch, "Loss", loss)

# Predictions
y_pred = tf.matmul(X, W) + b

# Plot results
plt.figure()
plt.scatter(X, y, label="Data")
plt.plot(X, y_pred.numpy(), label="Fitted Line")
plt.title("Linear Regression using tf.GradientTape")
plt.legend()
plt.show()
