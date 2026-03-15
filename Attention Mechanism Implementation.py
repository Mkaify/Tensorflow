import tensorflow as tf
import numpy as np
 
# Simulated encoder outputs (batch_size=1, time_steps=4, hidden_dim=8)
encoder_outputs = tf.random.normal([1, 4, 8])            # 4 encoder hidden states
decoder_hidden_state = tf.random.normal([1, 8])          # Current decoder hidden state
 
# Define basic attention layer
class BasicAttention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(8)               # Linear for encoder outputs
        self.W2 = tf.keras.layers.Dense(8)               # Linear for decoder state
        self.V = tf.keras.layers.Dense(1)                # Scoring layer
 
    def call(self, encoder_outputs, decoder_hidden):
        decoder_hidden_exp = tf.expand_dims(decoder_hidden, 1)          # Expand for broadcasting
        score = self.V(tf.nn.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden_exp)))  # Score eij
        attention_weights = tf.nn.softmax(score, axis=1)                # Normalize scores
        context_vector = attention_weights * encoder_outputs            # Weight encoder outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)          # Sum over time
        return context_vector, attention_weights
 
# Instantiate and compute attention
attention = BasicAttention()
context_vector, attention_weights = attention(encoder_outputs, decoder_hidden_state)
 
# Print shapes and example values
print("Encoder outputs shape:", encoder_outputs.shape)
print("Decoder hidden state shape:", decoder_hidden_state.shape)
print("Context vector shape:", context_vector.shape)
print("Attention weights shape:", attention_weights.shape)
print("Attention weights:", tf.squeeze(attention_weights).numpy())