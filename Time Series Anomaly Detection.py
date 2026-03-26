import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ Reproducibility
# ===============================
tf.keras.utils.set_random_seed(42)

# ===============================
# 2️⃣ Generate Synthetic Series
# ===============================
def generate_series(size=300):
    x = np.linspace(0, 50, size)
    y = np.sin(x) + np.random.normal(scale=0.1, size=size)
    y[80:85] += 3  # Inject anomaly
    y[180:185] -= 3
    return y.astype(np.float32)

series = generate_series()

# ===============================
# 3️⃣ Normalize
# ===============================
mean, std = series.mean(), series.std()
series_norm = (series - mean) / std

# ===============================
# 4️⃣ Create Sliding Window Sequences
# ===============================
def create_sequences(series, window=24):
    X = []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
    return np.array(X)[..., np.newaxis]  # Add channel dimension

window_size = 24
X = create_sequences(series_norm, window=window_size)

# ===============================
# 5️⃣ Train/Test Split
# ===============================
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]

# ===============================
# 6️⃣ Build tf.data Datasets
# ===============================
batch_size = 32

train_ds = tf.data.Dataset.from_tensor_slices((X_train, X_train))
train_ds = train_ds.shuffle(buffer_size=256).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, X_test))
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ===============================
# 7️⃣ Build LSTM Autoencoder
# ===============================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window_size, 1)),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.RepeatVector(window_size),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
])

model.summary()

# ===============================
# 8️⃣ Compile
# ===============================
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")

# ===============================
# 9️⃣ Callbacks
# ===============================
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

# ===============================
# 🔟 Train
# ===============================
history = model.fit(train_ds, epochs=30, validation_data=test_ds, callbacks=callbacks)

# ===============================
# 1️⃣1️⃣ Predict and Compute Reconstruction Error
# ===============================
X_pred = model.predict(test_ds)
# Flatten batches for comparison
X_test_flat = np.concatenate([y for x, y in test_ds], axis=0)
mse = np.mean(np.square(X_test_flat - X_pred), axis=(1, 2))

# ===============================
# 1️⃣2️⃣ Detect Anomalies
# ===============================
threshold = np.percentile(mse, 95)  # 95th percentile
anomalies = mse > threshold
anomaly_points = np.where(anomalies)[0] + train_size + window_size

# ===============================
# 1️⃣3️⃣ Plot Results
# ===============================
plt.figure(figsize=(12, 4))
plt.plot(series_norm, label="Normalized Series")
plt.scatter(anomaly_points, series_norm[anomaly_points], color="red", label="Anomalies")
plt.title("Time Series Anomaly Detection with LSTM Autoencoder")
plt.legend()
plt.show()