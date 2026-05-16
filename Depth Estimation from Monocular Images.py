import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

IMAGE_SIZE = (256, 256)


def load_and_preprocess(image_url):

    path = tf.keras.utils.get_file(origin=image_url)

    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, IMAGE_SIZE)

    img = tf.expand_dims(img, axis=0)  # NHWC

    return img


def normalize_depth(depth_map):
    depth_min = tf.reduce_min(depth_map)
    depth_max = tf.reduce_max(depth_map)
    return (depth_map - depth_min) / (depth_max - depth_min + 1e-8)


def main():

    model_url = "https://tfhub.dev/intel/midas/v2_1_small/1"

    image_url = (
        "https://storage.googleapis.com/download.tensorflow.org/"
        "example_images/YellowLabradorLooking_new.jpg"
    )

    # Load model
    depth_model = hub.load(model_url, tags=["serve"])
    infer = depth_model.signatures["serving_default"]

    # Load image (NHWC)
    input_image = load_and_preprocess(image_url)

    # 🔥 FIX: Convert NHWC → NCHW
    input_image_nchw = tf.transpose(input_image, [0, 3, 1, 2])

    # Inference
    outputs = infer(input_image_nchw)

    depth_tensor = list(outputs.values())[0]

    depth_map = depth_tensor[0]  # remove batch

    # Resize to display size
    depth_map = tf.image.resize(
        depth_map[..., tf.newaxis],
        IMAGE_SIZE
    )

    normalized_depth = normalize_depth(depth_map)

    # Display
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(tf.squeeze(input_image))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(tf.squeeze(normalized_depth), cmap="inferno")
    plt.title("Depth")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()