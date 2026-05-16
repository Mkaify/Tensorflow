import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np   # ✅ Missing import (important)

# Load a pretrained model from TensorFlow Hub
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

model = tf.keras.Sequential([
    hub.KerasLayer(model_url, input_shape=(224, 224, 3), trainable=False)
])

# Load ImageNet labels
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
)

imagenet_labels = np.array(open(labels_path).read().splitlines())

# Load sample image
image_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/puppy.jpg"

image_path = tf.keras.utils.get_file(
    "puppy.jpg",
    origin=image_url
)

# Preprocess image
def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    return tf.expand_dims(img, axis=0)

image = load_and_preprocess_image(image_path)

# Predict
predictions = model(image)

predicted_class = tf.argmax(predictions[0]).numpy()
predicted_label = imagenet_labels[predicted_class]

# Show result
plt.imshow(tf.squeeze(image))
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()
