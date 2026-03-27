import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 0️⃣ Suppress oneDNN warnings
# ===============================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ===============================
# 1️⃣ Reproducibility
# ===============================
tf.keras.utils.set_random_seed(42)

# ===============================
# 2️⃣ Generate synthetic data
# ===============================
def generate_series(size=500):
    x = np.linspace(0, 50, size)
    y = np.sin(x) + np.random.normal(scale=0.1, size=size)
    return y.astype(np.float32)

series = generate_series()

# ===============================
# 3️⃣ Normalize
# ===============================
mean, std = series.mean(), series.std()
series_norm = (series - mean) / std

# ===============================
# 4️⃣ Create input-output sequences
# ===============================
def create_dataset(series, window_size=30):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X)[..., np.newaxis], np.array(y)

window_size = 30
X, y = create_dataset(series_norm, window_size)

# ===============================
# 5️⃣ Train/Test Split
# ===============================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ===============================
# 6️⃣ Build tf.data pipelines
# ===============================
batch_size = 32

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(buffer_size=256).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ===============================
# 7️⃣ Build Conv1D Model
# ===============================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window_size, 1)),
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1)
])

model.summary()

# ===============================
# 8️⃣ Compile Model
# ===============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["mae"]
)

# ===============================
# 9️⃣ Callbacks
# ===============================
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

# ===============================
# 🔟 Train
# ===============================
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=30,
    callbacks=callbacks
)

# ===============================
# 1️⃣1️⃣ Predict
# ===============================
preds = model.predict(X_test[:100]).flatten()

# ===============================
# 1️⃣2️⃣ Plot
# ===============================
plt.figure(figsize=(10,4))
plt.plot(y_test[:100], label="True")
plt.plot(preds[:100], label="Predicted")
plt.title("Time Series Forecasting with Conv1D")
plt.xlabel("Step")
plt.ylabel("Normalized Value")
plt.legend()
plt.show()