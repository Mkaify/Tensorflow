import tensorflow as tf
import numpy as np

# -------------------------------------------------
# Sample parallel corpus
# -------------------------------------------------
english_sentences = ["hello", "how are you", "thank you", "good night"]
french_sentences = ["bonjour", "comment ça va", "merci", "bonne nuit"]

# -------------------------------------------------
# Tokenize Source (English)
# -------------------------------------------------
src_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
src_tokenizer.fit_on_texts(english_sentences)
src_sequences = src_tokenizer.texts_to_sequences(english_sentences)
src_vocab_size = len(src_tokenizer.word_index) + 1

# -------------------------------------------------
# Tokenize Target (French) with <start>, <end>
# -------------------------------------------------
french_sentences = [f"<start> {s} <end>" for s in french_sentences]

tgt_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
tgt_tokenizer.fit_on_texts(french_sentences)
tgt_sequences = tgt_tokenizer.texts_to_sequences(french_sentences)
tgt_vocab_size = len(tgt_tokenizer.word_index) + 1

# -------------------------------------------------
# Padding
# -------------------------------------------------
src_padded = tf.keras.preprocessing.sequence.pad_sequences(src_sequences, padding='post')
tgt_padded = tf.keras.preprocessing.sequence.pad_sequences(tgt_sequences, padding='post')

# Decoder input & output
decoder_input = tgt_padded[:, :-1]
decoder_target = tf.keras.utils.to_categorical(
    tgt_padded[:, 1:], num_classes=tgt_vocab_size
)

# -------------------------------------------------
# Build Seq2Seq Model
# -------------------------------------------------
embedding_dim = 64
latent_dim = 64

# Encoder
encoder_inputs = tf.keras.Input(shape=(None,))
enc_emb = tf.keras.layers.Embedding(src_vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = tf.keras.layers.LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = tf.keras.Input(shape=(None,))
dec_emb_layer = tf.keras.layers.Embedding(tgt_vocab_size, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

decoder_dense = tf.keras.layers.Dense(tgt_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Full training model
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------------------------------
# Train
# -------------------------------------------------
model.fit([src_padded, decoder_input], decoder_target, epochs=300, verbose=0)

# =================================================
# 🔥 INFERENCE MODELS
# =================================================

# Encoder inference model
encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

# Decoder inference model
decoder_state_input_h = tf.keras.Input(shape=(latent_dim,))
decoder_state_input_c = tf.keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2, initial_state=decoder_states_inputs
)

decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = tf.keras.Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2
)

# =================================================
# Translation Function
# =================================================

def translate(input_text):
    # Encode input
    seq = src_tokenizer.texts_to_sequences([input_text])
    seq = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=src_padded.shape[1], padding='post'
    )

    states = encoder_model.predict(seq, verbose=0)

    # Start token
    target_seq = np.array([[tgt_tokenizer.word_index['<start>']]])

    translated_sentence = []

    for _ in range(10):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states, verbose=0
        )

        token_index = np.argmax(output_tokens[0, -1, :])
        word = tgt_tokenizer.index_word.get(token_index, '')

        if word == '<end>' or word == '':
            break

        translated_sentence.append(word)

        target_seq = np.array([[token_index]])
        states = [h, c]

    return ' '.join(translated_sentence)

# -------------------------------------------------
# Test
# -------------------------------------------------
print("Translate 'thank you':", translate("thank you"))
print("Translate 'hello':", translate("hello"))