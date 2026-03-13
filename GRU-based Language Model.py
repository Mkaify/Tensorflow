import tensorflow as tf
import numpy as np

# -------------------------------------------------
# Tiny Shakespeare Text
# -------------------------------------------------
text = "To be, or not to be, that is the question."

# Build vocabulary
chars = sorted(set(text))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = np.array(chars)
vocab_size = len(chars)

# Convert text to integer sequence
text_as_int = np.array([char2idx[c] for c in text])

# -------------------------------------------------
# Create Training Sequences
# -------------------------------------------------
seq_length = 10
inputs = []
targets = []

for i in range(len(text_as_int) - seq_length):
    inputs.append(text_as_int[i:i+seq_length])
    targets.append(text_as_int[i+1:i+seq_length+1])

X = np.array(inputs)
y = np.array(targets)

# -------------------------------------------------
# Build GRU Language Model
# -------------------------------------------------
embedding_dim = 32
gru_units = 64

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.GRU(gru_units, return_sequences=True),
    tf.keras.layers.Dense(vocab_size)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.summary()

# -------------------------------------------------
# Train
# -------------------------------------------------
model.fit(X, y, epochs=200, verbose=0)

# =================================================
# 🔥 Text Generation with Temperature
# =================================================

def generate_text(seed, length=100, temperature=1.0):

    # Convert seed exactly as-is (case sensitive)
    input_eval = [char2idx[c] for c in seed]
    input_eval = tf.expand_dims(input_eval, 0)

    result = list(seed)

    for _ in range(length):

        predictions = model(input_eval)
        predictions = predictions[:, -1, :] / temperature

        predicted_id = tf.random.categorical(
            predictions, num_samples=1
        )[0, 0].numpy()

        result.append(idx2char[predicted_id])

        input_eval = tf.concat(
            [input_eval[:, 1:], [[predicted_id]]],
            axis=1
        )

    return ''.join(result)


# -------------------------------------------------
# Generate Text
# -------------------------------------------------
print(generate_text("To be, or ", length=80, temperature=0.8))