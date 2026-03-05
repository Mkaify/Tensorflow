import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# -----------------------------------
# Image Loading & Preprocessing
# -----------------------------------
def load_and_resize_image(url, scale=4, hr_size=(128, 128)):
    # Download image
    path = tf.keras.utils.get_file(origin=url)

    # Read & decode
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)

    # Convert to float32 [0,1]
    img = tf.image.convert_image_dtype(img, tf.float32)

    # High-resolution image
    hr_img = tf.image.resize(img, hr_size)

    # Downscale (simulate low-res)
    lr_small = tf.image.resize(
        hr_img,
        (hr_size[0] // scale, hr_size[1] // scale),
        method="bicubic"
    )

    # Upscale back to original size (blurry input)
    lr_img = tf.image.resize(lr_small, hr_size, method="bicubic")

    return lr_img, hr_img


# -----------------------------------
# Visualization
# -----------------------------------
def display_results(lr_img, hr_img, sr_img):
    plt.figure(figsize=(12, 4))

    def show(img, title):
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")

    plt.subplot(1, 3, 1)
    show(lr_img, "Low-Res Input")

    plt.subplot(1, 3, 2)
    show(hr_img, "Original High-Res")

    plt.subplot(1, 3, 3)
    show(sr_img, "Super-Res Output")

    plt.tight_layout()
    plt.show()


# -----------------------------------
# Main Execution
# -----------------------------------
def main():

    img_url = (
        "https://storage.googleapis.com/download.tensorflow.org/"
        "example_images/YellowLabradorLooking_new.jpg"
    )

    # Load images
    lr_img, hr_img = load_and_resize_image(img_url)

    # Add batch dimension
    lr_tensor = tf.expand_dims(lr_img, axis=0)

    # Load ESRGAN model from TF Hub
    model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")

    # Run super-resolution
    sr_tensor = model(lr_tensor)

    # Remove batch dim and clip
    sr_img = tf.clip_by_value(sr_tensor[0], 0.0, 1.0)

    # Convert tensors to numpy only for plotting
    display_results(
        lr_img.numpy(),
        hr_img.numpy(),
        sr_img.numpy()
    )


if __name__ == "__main__":
    main()