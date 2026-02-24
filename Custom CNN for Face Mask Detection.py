import tensorflow as tf
import matplotlib.pyplot as plt
import os


# ================================
# Dataset Directory (Change if Needed)
# ================================
DATASET_DIR = r"D:\Tensorflow\mask_dataset"


# ================================
# Parameters
# ================================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


# ================================
# Load Training Dataset
# ================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ================================
# Load Validation Dataset
# ================================
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ================================
# Performance Optimization
# ================================
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ================================
# Build CNN Model
# ================================
model = tf.keras.Sequential([

    tf.keras.Input(shape=(128, 128, 3)),

    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu"),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation="sigmoid")
])


# ================================
# Compile Model
# ================================
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ================================
# Train Model
# ================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    verbose=1
)


# ================================
# Prediction Demo
# ================================
for images, labels in val_ds.take(1):

    preds = model.predict(images)

    plt.imshow(images[0].numpy().astype("uint8"))

    label = "Mask" if preds[0] < 0.5 else "No Mask"

    plt.title(f"Predicted: {label}")

    plt.axis("off")
    plt.show()

    break
