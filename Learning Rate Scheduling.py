import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ================================
# Generate Data
# ================================
X = np.linspace(0, 10, 200).reshape(-1, 1).astype(np.float32)
y = 7 * X + 5 + np.random.normal(0, 2, size=X.shape)

# ================================
# Learning Rate Scheduler
# ================================
def scheduler(epoch, lr):
    if epoch > 0 and epoch % 20 == 0:
        return lr * 0.5
    return lr


# ================================
# Model
# ================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),          # Proper Input Layer
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])


# ================================
# Compile
# ================================
initial_lr = 0.01

optimizer = tf.keras.optimizers.Adam(
    learning_rate=initial_lr
)

model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)


# ================================
# Learning Rate Tracker
# ================================
class LRTracker(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        lr = float(
            tf.keras.backend.get_value(
                self.model.optimizer.learning_rate
            )
        )

        self.model.lr_history.append(lr)


model.lr_history = []


# ================================
# Callbacks
# ================================
lr_callback = tf.keras.callbacks.LearningRateScheduler(
    scheduler,
    verbose=0
)


# ================================
# Train
# ================================
history = model.fit(
    X, y,
    epochs=100,
    batch_size=32,
    callbacks=[lr_callback, LRTracker()],
    verbose=0
)


# ================================
# Predict
# ================================
preds = model.predict(X, verbose=0)


# ================================
# Plot Regression
# ================================
plt.figure(figsize=(8, 5))

plt.scatter(X, y, label="True Data", alpha=0.6)
plt.plot(X, preds, label="Predicted Line")

plt.title("Regression with Learning Rate Scheduling")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.show()


# ================================
# Plot Learning Rate
# ================================
plt.figure(figsize=(8, 5))

plt.plot(model.lr_history)

plt.title("Learning Rate Schedule During Training")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.grid(True)

plt.show()
