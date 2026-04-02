import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa

# ===============================
# 0️⃣ Suppress oneDNN warnings
# ===============================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ===============================
# 1️⃣ Reproducibility
# ===============================
tf.keras.utils.set_random_seed(42)

# ===============================
# 2️⃣ Load audio file
# ===============================
url = "https://github.com/pyannote/pyannote-audio/raw/develop/tutorials/assets/sample.wav"
path = tf.keras.utils.get_file("sample.wav", origin=url)

y, sr = librosa.load(path, sr=16000)                     # 16kHz sampling
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)      # 13 MFCCs
mfccs = mfccs.T                                          # Shape: (time, features)

# ===============================
# 3️⃣ Simulate dataset
# ===============================
num_samples = 100
X = np.stack([mfccs + np.random.normal(scale=0.1, size=mfccs.shape) for _ in range(num_samples)])
y_labels = np.random.randint(0, 3, size=num_samples)      # 3 emotion classes

# ===============================
# 4️⃣ Pad sequences
# ===============================
max_len = 100
X_padded = tf.keras.preprocessing.sequence.pad_sequences(
    X, maxlen=max_len, padding='post', dtype='float32'
)

# ===============================
# 5️⃣ One-hot labels
# ===============================
num_classes = 3
y_cat = tf.keras.utils.to_categorical(y_labels, num_classes=num_classes)

# ===============================
# 6️⃣ Train/test split
# ===============================
split = int(0.8 * num_samples)
X_train, X_test = X_padded[:split], X_padded[split:]
y_train, y_test = y_cat[:split], y_cat[split:]

# ===============================
# 7️⃣ Build DNN model
# ===============================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_len, 13)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

# ===============================
# 8️⃣ Compile
# ===============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)

# ===============================
# 9️⃣ Callbacks
# ===============================
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

# ===============================
# 🔟 Train with tf.data pipeline
# ===============================
batch_size = 16
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(64).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=30,
    callbacks=callbacks
)

# ===============================
# 1️⃣1️⃣ Predict a single example
# ===============================
pred = model.predict(X_padded[:1])[0]
emotion = int(np.argmax(pred))
print("Predicted Emotion Class:", emotion)