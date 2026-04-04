import os
import numpy as np
import tensorflow as tf
import librosa
import random

# ===============================
# 0️⃣ Reproducibility + Clean Logs
# ===============================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.keras.utils.set_random_seed(42)
np.random.seed(42)
random.seed(42)

# ===============================
# 1️⃣ Generate Simulated Urban Audio Data
# ===============================
def generate_urban_audio_data(classes=5,
                              samples_per_class=40,
                              max_len=100,
                              n_mfcc=20):

    X, y = [], []

    for label in range(classes):
        for _ in range(samples_per_class):

            # Slight frequency shift per class
            freq = random.uniform(200, 1000) + label * 100

            signal = np.sin(np.linspace(0, 2 * np.pi * freq, 22050))

            mfcc = librosa.feature.mfcc(
                y=signal.astype(np.float32),
                sr=22050,
                n_mfcc=n_mfcc
            )

            mfcc = mfcc.T[:max_len]

            # Padding
            if mfcc.shape[0] < max_len:
                pad = max_len - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode='constant')

            X.append(mfcc)
            y.append(label)

    X = np.array(X, dtype=np.float32)

    # Normalize MFCCs (VERY IMPORTANT in real projects)
    X = (X - np.mean(X)) / (np.std(X) + 1e-8)

    y = tf.keras.utils.to_categorical(y, num_classes=classes)

    return X, y


X, y = generate_urban_audio_data()

# Add channel dimension for CNN
X = X[..., np.newaxis]

print("Dataset shape:", X.shape)

# ===============================
# 2️⃣ Train/Test Split
# ===============================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ===============================
# 3️⃣ tf.data Pipeline
# ===============================
batch_size = 16

train_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .shuffle(500)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_test, y_test))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# ===============================
# 4️⃣ Build CNN Model
# ===============================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X.shape[1:]),

    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(y.shape[1], activation="softmax")
])

model.summary()

# ===============================
# 5️⃣ Compile
# ===============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)

# ===============================
# 6️⃣ Callbacks
# ===============================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    )
]

# ===============================
# 7️⃣ Train
# ===============================
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=30,
    callbacks=callbacks
)

# ===============================
# 8️⃣ Evaluate
# ===============================
loss, acc = model.evaluate(test_ds, verbose=0)
print(f"🏙️ Urban Sound Scene Classification Accuracy: {acc:.2f}")