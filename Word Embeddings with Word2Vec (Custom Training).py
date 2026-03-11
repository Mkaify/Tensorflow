import tensorflow as tf
import numpy as np

# -------------------------------------------------
# Sample corpus
# -------------------------------------------------
sentences = [
    "machine learning is fun",
    "deep learning is part of machine learning",
    "natural language processing is a field of ai",
    "word embeddings are learned representations",
    "tensorflow makes it easy to build models"
]

# -------------------------------------------------
# Tokenization
# -------------------------------------------------
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)

word2idx = tokenizer.word_index
idx2word = {v: k for k, v in word2idx.items()}
vocab_size = len(word2idx) + 1

print("Vocabulary Size:", vocab_size)

# -------------------------------------------------
# Generate Skip-gram pairs
# -------------------------------------------------
window_size = 2
sequences = tokenizer.texts_to_sequences(sentences)

pairs = []
for seq in sequences:
    for i, target_word in enumerate(seq):
        context_window = (
            seq[max(i - window_size, 0): i] +
            seq[i + 1: i + window_size + 1]
        )
        for context_word in context_window:
            pairs.append((target_word, context_word))

targets, contexts = zip(*pairs)
targets = np.array(targets)
contexts = np.array(contexts)

# -------------------------------------------------
# One-hot encode context
# -------------------------------------------------
context_labels = tf.keras.utils.to_categorical(
    contexts, num_classes=vocab_size
)

# -------------------------------------------------
# Define Skip-gram Model
# -------------------------------------------------
embedding_dim = 64

input_word = tf.keras.Input(shape=(1,))
embedding_layer = tf.keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    name="embedding"
)

embedding = embedding_layer(input_word)
x = tf.keras.layers.Reshape((embedding_dim,))(embedding)

output = tf.keras.layers.Dense(
    vocab_size,
    activation='softmax'
)(x)

model = tf.keras.Model(inputs=input_word, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy'
)

model.summary()

# -------------------------------------------------
# Train Model
# -------------------------------------------------
model.fit(
    targets,
    context_labels,
    epochs=200,
    verbose=0
)

# -------------------------------------------------
# Extract Learned Embeddings
# -------------------------------------------------
embedding_weights = model.get_layer("embedding").get_weights()[0]

print("\nLearned Word Embeddings (first 5 dimensions):\n")
for word, idx in word2idx.items():
    vec = embedding_weights[idx][:5]
    print(f"{word}: {np.round(vec, 3)}")

# -------------------------------------------------
# Cosine Similarity Test
# -------------------------------------------------
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )

print("\nCosine Similarity Examples:\n")

word1 = "machine"
word2 = "learning"

sim = cosine_similarity(
    embedding_weights[word2idx[word1]],
    embedding_weights[word2idx[word2]]
)

print(f"Similarity({word1}, {word2}) = {sim:.3f}")