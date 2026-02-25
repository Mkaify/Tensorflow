import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import layers, models


# ==============================
# Paths
# ==============================
DATASET_DIR = r"D:\Tensorflow\cats_dogs"


# ==============================
# Parameters
# ==============================
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
SEED = 42


# ==============================
# Load Dataset
# ==============================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ==============================
# Preprocess for InceptionV3
# ==============================
def preprocess(x, y):
    x = preprocess_input(x)
    return x, y


train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)


# ==============================
# Load Pretrained Model
# ==============================
base_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(299, 299, 3)
)

base_model.trainable = False   # Freeze initially


# ==============================
# Build Model
# ==============================
model = models.Sequential([

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])


# ==============================
# Compile (Phase 1)
# ==============================
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==============================
# Train (Frozen)
# ==============================
print("Training Frozen Model...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)


# ==============================
# Fine-Tuning
# ==============================
base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print("Fine-Tuning...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)


# ==============================
# Prediction Demo
# ==============================
for images, labels in val_ds.take(1):

    preds = model.predict(images)

    plt.imshow(images[0].numpy().astype("uint8"))

    label = "Dog" if preds[0] > 0.5 else "Cat"

    plt.title(f"Predicted: {label}")

    plt.axis("off")
    plt.show()

    break