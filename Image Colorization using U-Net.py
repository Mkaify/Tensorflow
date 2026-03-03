import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.color import rgb2lab, lab2rgb
from skimage.io import imread
from skimage.transform import resize


# ================================
# CONFIG
# ================================
IMG_SIZE = 128


# ================================
# DOWNLOAD SAMPLE IMAGE
# ================================
def download_image():

    url = "https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg"
    filename = "color_image.jpg"

    if not os.path.exists(filename):
        print("Downloading sample image...")
        tf.keras.utils.get_file(filename, origin=url)

    return filename


image_path = download_image()


# ================================
# LOAD & PREPROCESS IMAGE
# ================================
def preprocess_image(path):

    rgb = imread(path)

    rgb = resize(rgb, (IMG_SIZE, IMG_SIZE))

    rgb = rgb.astype(np.float32) / 255.0

    lab = rgb2lab(rgb)

    # Normalize
    L = lab[..., 0:1] / 100.0
    ab = lab[..., 1:] / 128.0

    # Add batch dimension
    L = L[np.newaxis, ...]
    ab = ab[np.newaxis, ...]

    return L, ab


X_l, y_ab = preprocess_image(image_path)


# ================================
# SIMPLE U-NET MODEL
# ================================
def build_model():

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Encoder
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Decoder
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)

    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)

    outputs = tf.keras.layers.Conv2D(
        2, 3, activation="tanh", padding="same"
    )(x)

    return tf.keras.Model(inputs, outputs)


model = build_model()

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()


# ================================
# TRAIN (DEMO PURPOSE)
# ================================
print("Training model (demo mode)...")

model.fit(
    X_l,
    y_ab,
    epochs=80,
    verbose=1
)


# ================================
# PREDICT COLOR
# ================================
pred_ab = model.predict(X_l)[0]


# ================================
# RECONSTRUCT IMAGE
# ================================
L = X_l[0] * 100.0
ab = pred_ab * 128.0

lab_out = np.concatenate([L, ab], axis=-1)

rgb_out = lab2rgb(lab_out)


# ================================
# DISPLAY RESULTS
# ================================
plt.figure(figsize=(10, 5))

# Input
plt.subplot(1, 2, 1)
plt.imshow(X_l[0].squeeze(), cmap="gray")
plt.title("Grayscale Input")
plt.axis("off")

# Output
plt.subplot(1, 2, 2)
plt.imshow(rgb_out)
plt.title("Colorized Output")
plt.axis("off")

plt.tight_layout()
plt.show()