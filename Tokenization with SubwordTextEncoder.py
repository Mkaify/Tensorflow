
import tensorflow as tf

# -----------------------------------
# Sample corpus
# -----------------------------------
corpus = [
    "TensorFlow is an end-to-end open-source platform for machine learning.",
    "Natural Language Processing is a fascinating field.",
    "Tokenization is the first step in NLP pipelines.",
    "Subword tokenization helps with rare words."
]

# -----------------------------------
# Build TextVectorization layer
# -----------------------------------
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=1000,
    output_mode="int",
    output_sequence_length=20,
    standardize="lower_and_strip_punctuation"
)

# Adapt on corpus
vectorizer.adapt(corpus)

# Vocabulary
vocab = vectorizer.get_vocabulary()

print("Vocabulary size:", len(vocab))

# -----------------------------------
# Encode sentence
# -----------------------------------
test_sentence = "Subword tokenization is powerful for text models."

encoded = vectorizer([test_sentence])

# Decode manually (since TextVectorization has no built-in decode)
decoded_tokens = [vocab[token] for token in encoded.numpy()[0] if token != 0]
decoded_sentence = " ".join(decoded_tokens)

# -----------------------------------
# Display Results
# -----------------------------------
print("\nOriginal Sentence:\n", test_sentence)

print("\nEncoded Tokens:\n", encoded.numpy()[0])

print("\nDecoded Approximation:\n", decoded_sentence)

print("\nIndividual Tokens:")
print(decoded_tokens)