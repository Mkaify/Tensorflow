import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import os


# ================================
# CONFIG
# ================================
IMAGE_SIZE = 513
MODEL_URL = "https://tfhub.dev/tensorflow/deeplabv3/1"


# ================================
# LOAD MODEL
# ================================
print("Loading DeepLabV3 Model...")

model = hub.load(MODEL_URL)

print("Model Loaded Successfully!")


# ================================
# DOWNLOAD SAMPLE IMAGE (SAFE WAY)
# ================================
def download_image():

    url = "https://upload.wikimedia.org/wikipedia/commons/5/5d/Street_scene_in_India.jpg"

    filename = "street.jpg"

    if not os.path.exists(filename):
        print("Downloading sample image...")
        tf.keras.utils.get_file(filename, origin=url)

    return filename


image_path = download_image()


# ================================
# IMAGE PREPROCESSING
# ================================
def preprocess_image(path):

    img_raw = tf.io.read_file(path)

    img = tf.image.decode_jpeg(img_raw, channels=3)

    original_size = tf.shape(img)[:2]

    # Resize
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # Normalize
    img = tf.cast(img, tf.float32) / 255.0

    # Add batch dimension
    img = tf.expand_dims(img, axis=0)

    return img, original_size, img_raw


input_image, original_size, raw_image = preprocess_image(image_path)


# ================================
# RUN SEGMENTATION
# ================================
print("Running Semantic Segmentation...")

result = model(input_image)

segmentation_map = tf.argmax(
    result["semantic_pred"],
    axis=-1
)[0]


# ================================
# PASCAL VOC COLORMAP
# ================================
def create_pascal_colormap():

    colormap = np.zeros((256, 3), dtype=np.uint8)

    for i in range(256):

        r = g = b = 0
        cid = i

        for j in range(8):

            r |= (cid & 1) << (7 - j)
            g |= ((cid >> 1) & 1) << (7 - j)
            b |= ((cid >> 2) & 1) << (7 - j)

            cid >>= 3

        colormap[i] = [r, g, b]

    return colormap


colormap = create_pascal_colormap()


# ================================
# COLORIZE SEGMENTATION MAP
# ================================
segmentation_color = colormap[segmentation_map.numpy()]

segmentation_color = tf.image.resize(
    segmentation_color,
    original_size,
    method="nearest"
)


# ================================
# DISPLAY RESULTS
# ================================
plt.figure(figsize=(12, 6))

# Original
plt.subplot(1, 2, 1)
plt.imshow(tf.image.decode_jpeg(raw_image))
plt.title("Original Image")
plt.axis("off")

# Segmented
plt.subplot(1, 2, 2)
plt.imshow(tf.cast(segmentation_color, tf.uint8))
plt.title("Semantic Segmentation")
plt.axis("off")

plt.tight_layout()
plt.show()