import tensorflow as tf
import numpy as np

# -----------------------------
# Sample tokenized sentence
# -----------------------------
sample_input = tf.constant([[3, 5, 7, 9, 0, 0]])  # Padded sequence of length 6
vocab_size = 20
maxlen = 6
embed_dim = 64
num_heads = 4
ff_dim = 128

# -----------------------------
# Positional Encoding Layer
# -----------------------------
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]

        # Compute position and dimension indices
        position = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)  # (seq_len, 1)
        i = tf.cast(tf.range(self.embed_dim)[tf.newaxis, :], tf.float32)  # (1, embed_dim)

        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(self.embed_dim, tf.float32))
        angle_rads = position * angle_rates  # (seq_len, embed_dim)

        # Apply sin to even indices and cos to odd indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        # Interleave sin and cos to match embedding dimension
        pos_encoding = tf.reshape(
            tf.concat([sines, cosines], axis=-1), (seq_len, self.embed_dim)
        )

        # Add batch dimension
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)

        return inputs + tf.cast(pos_encoding, tf.float32)

# -----------------------------
# Transformer Encoder Block
# -----------------------------
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        # Self-attention
        attn_output = self.att(x, x)
        out1 = self.norm1(x + attn_output)  # Add & norm
        ffn_output = self.ffn(out1)         # Feed-forward network
        return self.norm2(out1 + ffn_output)  # Add & norm

# -----------------------------
# Build the full encoder model
# -----------------------------
inputs = tf.keras.Input(shape=(maxlen,))
x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
x = PositionalEncoding(embed_dim)(x)
x = TransformerEncoder(embed_dim, num_heads, ff_dim)(x)

model = tf.keras.Model(inputs, x)

# -----------------------------
# Run the model on the sample input
# -----------------------------
output_embeddings = model(sample_input)

print("Output Embedding Shape:", output_embeddings.shape)
print("Token Embedding for first token (first 5 dims):", output_embeddings[0, 0, :5].numpy().round(3))