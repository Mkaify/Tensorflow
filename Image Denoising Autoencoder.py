import tensorflow as tf
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 128
EPOCHS = 5
NOISE_FACTOR = 0.5


# -----------------------------------
# Dataset Preparation
# -----------------------------------
def load_mnist():

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    # Convert to float32 and normalize
    x_train = tf.cast(x_train, tf.float32) / 255.0
    x_test = tf.cast(x_test, tf.float32) / 255.0

    # Add channel dimension
    x_train = tf.expand_dims(x_train, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)

    return x_train, x_test


def add_noise(images):
    noise = tf.random.normal(shape=tf.shape(images))
    noisy_images = images + NOISE_FACTOR * noise
    return tf.clip_by_value(noisy_images, 0.0, 1.0)


def prepare_dataset(x_train, x_test):

    train_ds = (
        tf.data.Dataset.from_tensor_slices(x_train)
        .map(lambda x: (add_noise(x), x), num_parallel_calls=AUTOTUNE)
        .shuffle(10000)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices(x_test)
        .map(lambda x: (add_noise(x), x), num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    return train_ds, test_ds


# -----------------------------------
# Model Definition
# -----------------------------------
def build_autoencoder():

    inputs = tf.keras.Input(shape=(28, 28, 1))

    # Encoder
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D(2, padding="same")(x)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(2, padding="same")(x)

    # Decoder
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D(2)(x)

    outputs = tf.keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# -----------------------------------
# Visualization
# -----------------------------------
def display_results(model, x_test):

    noisy_images = add_noise(x_test[:10])
    decoded_imgs = model.predict(noisy_images, verbose=0)

    plt.figure(figsize=(20, 4))

    for i in range(10):

        # Noisy
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(tf.squeeze(noisy_images[i]), cmap="gray")
        plt.title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(2, 10, i + 11)
        plt.imshow(tf.squeeze(decoded_imgs[i]), cmap="gray")
        plt.title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------------------
# Main
# -----------------------------------
def main():

    x_train, x_test = load_mnist()

    train_ds, test_ds = prepare_dataset(x_train, x_test)

    model = build_autoencoder()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["mse"]
    )

    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS
    )

    display_results(model, x_test)


if __name__ == "__main__":
    main()