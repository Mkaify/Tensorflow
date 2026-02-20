import tensorflow as tf
import matplotlib.pyplot as plt


# ================================
# Local Image Path
# ================================
image_path = r"D:\Tensorflow\images\dog.jpg"


# ================================
# Preprocess Function
# ================================
def preprocess_image(path):
    img_raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return img


# ================================
# Load Image
# ================================
image = preprocess_image(image_path)


# ================================
# Display
# ================================
plt.imshow(image)
plt.title("Preprocessed Image (Offline)")
plt.axis("off")
plt.show()
