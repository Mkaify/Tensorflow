import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# -----------------------------------
# Image Loading & Preprocessing
# -----------------------------------
def load_image_from_url(url, target_size=(256, 256)):
    # Download file
    path = tf.keras.utils.get_file(origin=url)

    # Read and decode image
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)

    # Convert to float32 in range [0,1]
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Resize
    img = tf.image.resize(img, target_size)

    # Add batch dimension
    img = tf.expand_dims(img, axis=0)

    return img


# -----------------------------------
# Image URLs
# -----------------------------------
content_url = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "example_images/YellowLabradorLooking_new.jpg"
)

style_url = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg"
)


# -----------------------------------
# Main Execution
# -----------------------------------
def main():

    # Load images
    content_image = load_image_from_url(content_url)
    style_image = load_image_from_url(style_url)

    # Load TensorFlow Hub model
    model = hub.load(
        "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    )

    # Apply style transfer
    stylized_image = model(content_image, style_image)[0]

    # Display results
    display_images(content_image, style_image, stylized_image)


# -----------------------------------
# Visualization
# -----------------------------------
def display_images(content, style, stylized):

    def show_image(img_tensor, title):
        img = tf.squeeze(img_tensor, axis=0)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    show_image(content, "Content")

    plt.subplot(1, 3, 2)
    show_image(style, "Style")

    plt.subplot(1, 3, 3)
    show_image(stylized, "Stylized")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()