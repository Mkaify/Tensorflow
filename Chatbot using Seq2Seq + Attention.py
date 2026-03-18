import tensorflow as tf
import numpy as np

# ----------------------------
# Sample dataset
# ----------------------------
questions = ["hi", "how are you", "what's your name", "bye"]
answers = ["hello", "i'm fine", "i'm a chatbot", "goodbye"]

answers = [f"<start> {a} <end>" for a in answers]

# ----------------------------
# Tokenization
# ----------------------------
q_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
a_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

q_tokenizer.fit_on_texts(questions)
a_tokenizer.fit_on_texts(answers)

q_seq = q_tokenizer.texts_to_sequences(questions)
a_seq = a_tokenizer.texts_to_sequences(answers)

max_q_len = max(len(q) for q in q_seq)
max_a_len = max(len(a) for a in a_seq)

q_pad = tf.keras.preprocessing.sequence.pad_sequences(q_seq, maxlen=max_q_len, padding='post')
a_pad = tf.keras.preprocessing.sequence.pad_sequences(a_seq, maxlen=max_a_len, padding='post')

decoder_input = a_pad[:, :-1]
decoder_target = tf.keras.utils.to_categorical(
    a_pad[:, 1:], num_classes=len(a_tokenizer.word_index) + 1
)

# ----------------------------
# Attention Layer
# ----------------------------
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, enc_output, dec_hidden):
        dec_hidden = tf.expand_dims(dec_hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(dec_hidden)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context = attention_weights * enc_output
        context_vector = tf.reduce_sum(context, axis=1)
        return context_vector

# ----------------------------
# Encoder
# ----------------------------
encoder_inputs = tf.keras.Input(shape=(max_q_len,))
enc_emb = tf.keras.layers.Embedding(len(q_tokenizer.word_index) + 1, 64)(encoder_inputs)
encoder_lstm = tf.keras.layers.LSTM(64, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

# ----------------------------
# Decoder
# ----------------------------
decoder_inputs = tf.keras.Input(shape=(max_a_len - 1,))
dec_emb_layer = tf.keras.layers.Embedding(len(a_tokenizer.word_index) + 1, 64)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = tf.keras.layers.LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

# Attention
attention = BahdanauAttention(64)
context_vector = attention(encoder_outputs, state_h)
context_vector = tf.expand_dims(context_vector, 1)
context_vector = tf.repeat(context_vector, tf.shape(decoder_outputs)[1], axis=1)

concat = tf.concat([decoder_outputs, context_vector], axis=-1)

final_dense = tf.keras.layers.Dense(len(a_tokenizer.word_index) + 1, activation='softmax')
final_output = final_dense(concat)

# ----------------------------
# Training model
# ----------------------------
model = tf.keras.Model([encoder_inputs, decoder_inputs], final_output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([q_pad, decoder_input], decoder_target, epochs=500, verbose=0)

# ============================
# INFERENCE MODELS
# ============================

# Encoder model for inference
encoder_model = tf.keras.Model(encoder_inputs, [encoder_outputs, state_h, state_c])

# Decoder inputs for inference (one token at a time)
dec_input_token = tf.keras.Input(shape=(1,))
enc_out_inf = tf.keras.Input(shape=(max_q_len, 64))
state_h_inf = tf.keras.Input(shape=(64,))
state_c_inf = tf.keras.Input(shape=(64,))

dec_emb_inf = dec_emb_layer(dec_input_token)

dec_outputs_inf, state_h_new, state_c_new = decoder_lstm(
    dec_emb_inf, initial_state=[state_h_inf, state_c_inf]
)

context_inf = attention(enc_out_inf, state_h_new)
context_inf = tf.expand_dims(context_inf, 1)

concat_inf = tf.concat([dec_outputs_inf, context_inf], axis=-1)
final_out_inf = final_dense(concat_inf)

decoder_model = tf.keras.Model(
    [dec_input_token, enc_out_inf, state_h_inf, state_c_inf],
    [final_out_inf, state_h_new, state_c_new]
)

# ----------------------------
# Chat Function
# ----------------------------
def chat(input_text):
    seq = q_tokenizer.texts_to_sequences([input_text])
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_q_len, padding='post')

    enc_out, h, c = encoder_model.predict(seq)

    dec_input = tf.constant([[a_tokenizer.word_index['<start>']]])
    result = []

    for _ in range(max_a_len):
        pred, h, c = decoder_model.predict([dec_input, enc_out, h, c])
        token = np.argmax(pred[0, 0, :])

        if a_tokenizer.index_word.get(token) == '<end>':
            break

        result.append(a_tokenizer.index_word.get(token))
        dec_input = tf.constant([[token]])

    return ' '.join(result)

# ----------------------------
# Test
# ----------------------------
print("User: hi")
print("Bot:", chat("hi"))