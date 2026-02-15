import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ================================
# Set Random Seeds
# ================================
np.random.seed(42)
tf.random.set_seed(42)


# ================================
# Download Example Image
# ================================
image_path = tf.keras.utils.get_file(
    "grace_hopper.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
)


# ================================
# Preprocessing Function
# ================================
def preprocess_image(path, target_size=(224, 224)):

    # Read image
    img_raw = tf.io.read_file(path)

    # Decode JPEG
    img = tf.image.decode_jpeg(img_raw, channels=3)

    # Resize
    img = tf.image.resize(img, target_size)

    # Convert to float
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Normalize (ImageNet-style mean/std)
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])

    img = (img - mean) / std

    return img


# ================================
# Preprocess Image
# ================================
image = preprocess_image(image_path)


# ================================
# Add Batch Dimension (For Models)
# ================================
image_batch = tf.expand_dims(image, axis=0)


# ================================
# Display Image (De-normalized)
# ================================
def denormalize(img):

    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])

    img = (img * std) + mean
    img = tf.clip_by_value(img, 0.0, 1.0)

    return img


display_img = denormalize(image)


plt.figure(figsize=(5, 5))

plt.imshow(display_img)
plt.title("Preprocessed & Normalized Image")
plt.axis("off")

plt.show()


print("Image batch shape:", image_batch.shape)
