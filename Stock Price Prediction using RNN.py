import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ Reproducibility
# ===============================
tf.keras.utils.set_random_seed(42)

# ===============================
# 2️⃣ Load Dataset Properly
# ===============================
#url = "https://raw.githubusercontent.com/selva86/datasets/master/aapl.csv"
#path = tf.keras.utils.get_file("aapl.csv", origin=url)
df = pd.read_csv("datasets/aapl.csv")

prices = df["Close"].astype("float32").values
prices = (prices - prices.mean()) / prices.std()

# ===============================
# 3️⃣ Create tf.data Dataset
# ===============================
window_size = 30
batch_size = 32

dataset = tf.keras.utils.timeseries_dataset_from_array(
    data=prices[:-window_size],
    targets=prices[window_size:],
    sequence_length=window_size,
    batch_size=batch_size,
    shuffle=True,
)

# Split into train/test
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)

train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size)

# ===============================
# 4️⃣ Build Model (Modern Style)
# ===============================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window_size, 1)),
    tf.keras.layers.SimpleRNN(64, activation="tanh"),
    tf.keras.layers.Dense(1)
])

model.summary()

# ===============================
# 5️⃣ Compile
# ===============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["mae"]
)

# ===============================
# 6️⃣ Callbacks (Best Practice)
# ===============================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        save_best_only=True
    )
]

# ===============================
# 7️⃣ Train
# ===============================
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50,
    callbacks=callbacks
)

# ===============================
# 8️⃣ Predict
# ===============================
preds = model.predict(test_ds)
preds = preds.flatten()

# Extract true values
true = np.concatenate([y for x, y in test_ds], axis=0)

# ===============================
# 9️⃣ Plot
# ===============================
plt.figure(figsize=(10,5))
plt.plot(true[:100], label="True Prices")
plt.plot(preds[:100], label="Predicted Prices")
plt.title("Stock Price Prediction with RNN")
plt.xlabel("Days")
plt.ylabel("Normalized Price")
plt.legend()
plt.show()