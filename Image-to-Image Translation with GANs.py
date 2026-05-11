import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape
from tensorflow.keras.models import Model

# -----------------------------
# Generator Model
# -----------------------------
def build_generator():
    model = tf.keras.Sequential([
        Input(shape=(256, 256, 3)),
        Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(256 * 256 * 3, activation='tanh'),  # Output values in [-1,1]
        Reshape((256, 256, 3))
    ])
    return model

# -----------------------------
# Discriminator Model
# -----------------------------
def build_discriminator():
    model = tf.keras.Sequential([
        Input(shape=(256, 256, 3)),
        Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        Flatten(),
        Dense(1, activation='sigmoid')  # Output probability: real or fake
    ])
    return model

# -----------------------------
# Build GAN
# -----------------------------
generator = build_generator()
discriminator = build_discriminator()

# Compile discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# GAN pipeline (generator + frozen discriminator)
input_img = Input(shape=(256, 256, 3))
generated_img = generator(input_img)
discriminator.trainable = False  # Freeze discriminator while training generator
validity = discriminator(generated_img)

gan_model = Model(input_img, validity)
gan_model.compile(optimizer='adam', loss='binary_crossentropy')

# -----------------------------
# Generate a sample image
# -----------------------------
sample_input = np.random.randn(1, 256, 256, 3)  # Random noise (replace with actual images)
generated_img = generator.predict(sample_input)

# Plot the generated image
plt.imshow((generated_img[0] + 1) / 2)  # Rescale from [-1,1] to [0,1] for display
plt.title("Generated Image (Image-to-Image Translation)")
plt.axis('off')
plt.show()