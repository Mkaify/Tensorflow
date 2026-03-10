import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------------------------------------
# Optional: Prevent GPU memory overflow
# -------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# -------------------------------------------------
# Load IMDB Dataset
# -------------------------------------------------
vocab_size = 10000

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=vocab_size
)

# -------------------------------------------------
# Pad sequences to fixed length
# -------------------------------------------------
maxlen = 300

X_train = tf.keras.preprocessing.sequence.pad_sequences(
    X_train,
    maxlen=maxlen,
    padding='post',
    truncating='post'
)

X_test = tf.keras.preprocessing.sequence.pad_sequences(
    X_test,
    maxlen=maxlen,
    padding='post',
    truncating='post'
)

# -------------------------------------------------
# Build Bi-LSTM Model
# -------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=128
    ),

    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64)
    ),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

# -------------------------------------------------
# Compile Model
# -------------------------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Build model (important for summary display)
model.build(input_shape=(None, maxlen))
model.summary()

# -------------------------------------------------
# Train Model
# -------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)

# -------------------------------------------------
# Evaluate Model
# -------------------------------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}")

# -------------------------------------------------
# Plot Accuracy & Loss
# -------------------------------------------------
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])

plt.tight_layout()
plt.show()