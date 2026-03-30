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
# 2️⃣ Generate synthetic air quality data
# ===============================
def generate_air_data(size=500):
    t = np.linspace(0, 50, size)
    pm25 = 50 + 10 * np.sin(0.3 * t) + np.random.normal(0, 5, size)
    no2 = 30 + 5 * np.cos(0.2 * t) + np.random.normal(0, 3, size)
    co = 1 + 0.2 * np.sin(0.1 * t) + np.random.normal(0, 0.1, size)
    aqi = 0.5 * pm25 + 0.3 * no2 + 20 * co + np.random.normal(0, 2, size)
    return np.stack([pm25, no2, co, aqi], axis=1).astype(np.float32)

data = generate_air_data()

# ===============================
# 3️⃣ Normalize features
# ===============================
mean = data.mean(axis=0)
std = data.std(axis=0)
data_norm = (data - mean) / std

# ===============================
# 4️⃣ Create sequences
# ===============================
def create_dataset(data, window=24, target_index=3):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window, target_index])
    return np.array(X), np.array(y)

window_size = 24
X, y = create_dataset(data_norm, window=window_size, target_index=3)

# ===============================
# 5️⃣ Train/Test split
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
# 7️⃣ Build LSTM model
# ===============================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window_size, X.shape[2])),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1)
])

model.summary()

# ===============================
# 8️⃣ Compile
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
# 1️⃣1️⃣ Predict & Plot
# ===============================
preds = model.predict(X_test[:100]).flatten()

plt.figure(figsize=(10,4))
plt.plot(y_test[:100], label="True AQI")
plt.plot(preds[:100], label="Predicted AQI")
plt.title("Air Quality Prediction (AQI)")
plt.xlabel("Hour")
plt.ylabel("Normalized AQI")
plt.legend()
plt.show()