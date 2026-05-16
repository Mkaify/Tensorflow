import tensorflow as tf
import numpy as np
 
# Sample token-level NER dataset
sentences = [["john", "lives", "in", "new", "york"],
             ["alice", "is", "from", "paris"],
             ["bob", "visited", "london", "last", "year"]]
 
labels = [["PER", "O", "O", "LOC", "LOC"],
          ["PER", "O", "O", "LOC"],
          ["PER", "O", "LOC", "O", "O"]]
 
# Build vocabularies
word_tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True, oov_token='UNK')
word_tokenizer.fit_on_texts(sentences)
X = word_tokenizer.texts_to_sequences(sentences)
word_index = word_tokenizer.word_index
vocab_size = len(word_index) + 1
 
tag_tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=False)
tag_tokenizer.fit_on_texts(labels)
y = tag_tokenizer.texts_to_sequences(labels)
tag_index = tag_tokenizer.word_index
num_tags = len(tag_index) + 1
 
# Pad sequences
max_len = max(len(s) for s in X)
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len, padding='post')
y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=max_len, padding='post')
 
# Convert labels to categorical
y_cat = tf.keras.utils.to_categorical(y, num_classes=num_tags)
 
# Build Bi-LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),  # Word embeddings
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),        # Bi-LSTM
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_tags, activation='softmax')) # One output per token
])
 
# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
# Train
model.fit(X, y_cat, epochs=50, verbose=0)
 
# Predict on a new sentence
test_sentence = ["alice", "visited", "new", "delhi"]
test_seq = word_tokenizer.texts_to_sequences([test_sentence])
test_seq = tf.keras.preprocessing.sequence.pad_sequences(test_seq, maxlen=max_len, padding='post')
 
pred = model.predict(test_seq)[0]
pred_tags = [list(tag_index.keys())[np.argmax(p) - 1] if np.argmax(p) > 0 else "PAD" for p in pred]
 
# Print prediction
for word, tag in zip(test_sentence, pred_tags):
    print(f"{word} → {tag}")