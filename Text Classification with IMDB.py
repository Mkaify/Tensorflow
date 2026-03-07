import tensorflow as tf
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE
VOCAB_SIZE = 10000
MAX_LEN = 200
BATCH_SIZE = 64
EPOCHS = 3


# -----------------------------------
# Dataset Preparation
# -----------------------------------
def load_imdb():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=VOCAB_SIZE
    )

    # Modern padding (recommended API)
    x_train = tf.keras.utils.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = tf.keras.utils.pad_sequences(x_test, maxlen=MAX_LEN)

    return (x_train, y_train), (x_test, y_test)


def prepare_datasets(x_train, y_train, x_test, y_test):

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(10000)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    return train_ds, test_ds


# -----------------------------------
# Model Definition
# -----------------------------------
def build_model():

    inputs = tf.keras.Input(shape=(MAX_LEN,))

    x = tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=64
    )(inputs)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64)
    )(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# -----------------------------------
# Visualization
# -----------------------------------
def plot_history(history):

    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")

    plt.title("IMDB Sentiment Classification")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------------
# Main
# -----------------------------------
def main():

    (x_train, y_train), (x_test, y_test) = load_imdb()

    train_ds, test_ds = prepare_datasets(
        x_train, y_train, x_test, y_test
    )

    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Accuracy: {acc:.4f}")

    plot_history(history)


if __name__ == "__main__":
    main()