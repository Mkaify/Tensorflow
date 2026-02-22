import tensorflow as tf
import matplotlib.pyplot as plt


# ================================
# Local Image Path
# ================================
image_path = r"D:\Tensorflow\images\dog.jpg"


# ================================
# Load & Preprocess Image
# ================================
img_raw = tf.io.read_file(image_path)

img = tf.image.decode_jpeg(img_raw, channels=3)

img = tf.image.resize(img, [224, 224])

img = tf.cast(img, tf.float32) / 255.0


# ================================
# Apply Augmentations
# ================================
augmented_images = [

    img,                                    # Original

    tf.image.flip_left_right(img),          # Horizontal Flip

    tf.image.flip_up_down(img),             # Vertical Flip

    tf.image.rot90(img),                    # Rotation

    tf.image.adjust_brightness(img, 0.3),   # Brightness

    tf.image.adjust_contrast(img, 2.0),     # Contrast

    tf.image.random_crop(
        tf.image.resize(img, [256, 256]),
        size=[224, 224, 3]
    )                                       # Random Crop
]


# ================================
# Visualization
# ================================
plt.figure(figsize=(12, 6))

for i, augmented in enumerate(augmented_images):

    plt.subplot(2, 4, i + 1)

    plt.imshow(tf.clip_by_value(augmented, 0.0, 1.0))

    plt.axis("off")

    plt.title(f"Augmentation {i+1}")


plt.tight_layout()
plt.show()
