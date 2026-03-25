import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ Reproducibility
# ===============================
tf.keras.utils.set_random_seed(42)

# ===============================
# 2️⃣ Load Dataset (Official URL)
# ===============================
#url = "https://storage.googleapis.com/download.tensorflow.org/data/jena_climate_2009_2016.csv"
#path = tf.keras.utils.get_file("jena_climate.csv", origin=url)
df = pd.read_csv("datasets/jena_climate_2009_2016.csv")

# Select features
feature_columns = ["T (degC)", "p (mbar)", "rh (%)", "wv (m/s)"]
data = df[feature_columns].astype("float32").values

# ===============================
# 3️⃣ Train/Test Split FIRST
# ===============================
split = int(0.8 * len(data))
train_data = data[:split]
test_data = data[split:]

# ===============================
# 4️⃣ Normalization Layer (Modern Way)
# ===============================
normalizer = tf.keras.layers.Normalization()
normalizer.adapt(train_data)

train_data = normalizer(train_data)
test_data = normalizer(test_data)

# ===============================
# 5️⃣ Create tf.data TimeSeries Dataset
# ===============================
window_size = 24
batch_size = 32
target_index = 0  # Temperature

def make_dataset(dataset):
    return tf.keras.utils.timeseries_dataset_from_array(
        data=dataset[:-window_size],
        targets=dataset[window_size:, target_index],
        sequence_length=window_size,
        batch_size=batch_size,
        shuffle=True
    )

train_ds = make_dataset(train_data)
test_ds = make_dataset(test_data)

# Optimize pipeline
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

# ===============================
# 6️⃣ Build Modern LSTM Model
# ===============================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window_size, len(feature_columns))),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1)
])

model.summary()

# ===============================
# 7️⃣ Compile
# ===============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["mae"]
)

# ===============================
# 8️⃣ Callbacks
# ===============================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_jena_model.keras",
        save_best_only=True
    )
]

# ===============================
# 9️⃣ Train
# ===============================
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=30,
    callbacks=callbacks
)

# ===============================
# 🔟 Predict & Plot
# ===============================
preds = model.predict(test_ds)
preds = preds.flatten()

true = np.concatenate([y for x, y in test_ds], axis=0)

plt.figure(figsize=(10,5))
plt.plot(true[:100], label="True Temp")
plt.plot(preds[:100], label="Predicted Temp")
plt.title("Multivariate Time Series Forecasting (LSTM)")
plt.xlabel("Hour")
plt.ylabel("Normalized Temperature")
plt.legend()
plt.show()