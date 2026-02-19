import tensorflow as tf
import tensorflow_datasets as tfds


# ================================
# Load CIFAR-10 Dataset
# ================================
(ds_train, ds_test), ds_info = tfds.load(
    "cifar10",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)


# ================================
# Preprocessing Function
# ================================
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0   # Normalize to [0,1]
    return image, label


# ================================
# Dataset Pipeline
# ================================
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

ds_train = (
    ds_train
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

ds_test = (
    ds_test
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


# ================================
# Build CNN Model
# ================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])


# ================================
# Compile Model
# ================================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ================================
# Train Model
# ================================
history = model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test,
    verbose=1
)


# ================================
# Evaluate Model
# ================================
test_loss, test_acc = model.evaluate(ds_test, verbose=0)
print(f"Test Accuracy: {test_acc:.3f}")
