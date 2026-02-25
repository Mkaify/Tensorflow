import tensorflow as tf
import matplotlib.pyplot as plt
import os


# ================================
# Dataset Path
# ================================
DATASET_DIR = r"D:\Tensorflow\cats_dogs"


# ================================
# Parameters
# ================================
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42


# ================================
# Load Dataset
# ================================
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


# ================================
# Normalize
# ================================
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ================================
# Load Pretrained MobileNetV2
# ================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False   # Freeze base model


# ================================
# Build Model
# ================================
model = tf.keras.Sequential([

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(1, activation="sigmoid")
])


# ================================
# Compile
# ================================
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ================================
# Train
# ================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    verbose=1
)


# ================================
# Prediction Demo
# ================================
for images, labels in val_ds.take(1):

    preds = model.predict(images)

    plt.imshow(images[0].numpy().astype("uint8"))

    label = "Dog" if preds[0] > 0.5 else "Cat"

    plt.title(f"Predicted: {label}")

    plt.axis("off")
    plt.show()

    break
