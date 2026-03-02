import tensorflow as tf
import numpy as np

# ===============================
# CONFIGURATION
# ===============================
BATCH_SIZE = 16
IMG_WIDTH = 100
IMG_HEIGHT = 32
MAX_LABEL_LENGTH = 10

# Characters: a-z + blank
CHARACTERS = "abcdefghijklmnopqrstuvwxyz"
NUM_CLASSES = len(CHARACTERS) + 1  # +1 for CTC blank


# ===============================
# DUMMY DATA GENERATOR (FOR DEMO)
# ===============================
def generate_dummy_data(batch_size):
    """
    Generate fake OCR images and labels (for demo only)
    """

    # Random grayscale images
    X = np.random.rand(
        batch_size,
        IMG_HEIGHT,
        IMG_WIDTH,
        1
    ).astype(np.float32)

    # Random labels (a-z -> 1 to 26)
    y = np.random.randint(
        1,
        len(CHARACTERS) + 1,
        size=(batch_size, MAX_LABEL_LENGTH)
    )

    # Length info for CTC
    input_length = np.ones((batch_size, 1)) * (IMG_WIDTH // 4)
    label_length = np.ones((batch_size, 1)) * MAX_LABEL_LENGTH

    return X, y, input_length, label_length


# ===============================
# BUILD OCR MODEL
# ===============================
def build_ocr_model():

    # Image input
    input_img = tf.keras.Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, 1),
        name="input_image"
    )

    # CNN Feature Extractor
    x = tf.keras.layers.Conv2D(
        32, (3, 3),
        activation="relu",
        padding="same"
    )(input_img)

    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(
        64, (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Reshape for RNN
    new_shape = (IMG_WIDTH // 4, (IMG_HEIGHT // 4) * 64)

    x = tf.keras.layers.Reshape(
        target_shape=new_shape
    )(x)

    # Bidirectional LSTM
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128,
            return_sequences=True
        )
    )(x)

    # Character prediction
    x = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    # CTC Inputs
    labels = tf.keras.Input(
        name="labels",
        shape=(None,),
        dtype="int32"
    )

    input_length = tf.keras.Input(
        name="input_length",
        shape=(1,),
        dtype="int32"
    )

    label_length = tf.keras.Input(
        name="label_length",
        shape=(1,),
        dtype="int32"
    )

    # CTC Loss Function
    def ctc_loss(args):
        y_pred, labels, input_len, label_len = args

        return tf.keras.backend.ctc_batch_cost(
            labels,
            y_pred,
            input_len,
            label_len
        )

    # CTC Loss Layer
    loss_out = tf.keras.layers.Lambda(
        ctc_loss,
        name="ctc_loss"
    )([x, labels, input_length, label_length])

    # Final Model
    model = tf.keras.Model(
        inputs=[input_img, labels, input_length, label_length],
        outputs=loss_out
    )

    return model


# ===============================
# TRAINING FUNCTION
# ===============================
def train_model(model, epochs=5):

    for epoch in range(epochs):

        # Generate dummy batch
        X, y, in_len, lab_len = generate_dummy_data(BATCH_SIZE)

        # Dummy output (CTC computes real loss)
        dummy_y = np.zeros((BATCH_SIZE, 1))

        loss = model.train_on_batch(
            {
                "input_image": X,
                "labels": y,
                "input_length": in_len,
                "label_length": lab_len
            },
            dummy_y
        )

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")


# ===============================
# MAIN FUNCTION
# ===============================
def main():

    print("Building OCR Model...")

    model = build_ocr_model()

    model.compile(
        optimizer="adam",
        loss=lambda y_true, y_pred: y_pred
    )

    model.summary()

    print("\nStarting Training...\n")

    train_model(model, epochs=5)

    print("\nTraining Completed Successfully!")


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    main()