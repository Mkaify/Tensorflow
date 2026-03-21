import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

# ----------------------------
# Vocabulary
# ----------------------------
vocab = ["economy", "money", "market",
         "football", "goal", "team",
         "python", "code", "model"]

vocab_size = len(vocab)

# ----------------------------
# Topic-word distributions
# (3 topics x 9 words)
# ----------------------------
topics = tf.constant([
    [0.4, 0.3, 0.3, 0,   0,   0,   0,   0,   0],  # Economy
    [0,   0,   0,   0.3, 0.4, 0.3, 0,   0,   0],  # Sports
    [0,   0,   0,   0,   0,   0,   0.3, 0.3, 0.4] # Tech
], dtype=tf.float32)

num_topics = 3
num_docs = 5
words_per_doc = 6

# ----------------------------
# Dirichlet prior for documents
# ----------------------------
doc_topic_dist = tfd.Dirichlet(concentration=[0.5] * num_topics)

# Sample topic mixtures for each document
theta = doc_topic_dist.sample(num_docs)  # shape (num_docs, 3)

# ----------------------------
# Generate documents
# ----------------------------
for d in range(num_docs):
    print(f"\nDocument {d+1}")
    print("Topic mixture:", theta[d].numpy().round(3))

    words = []

    for _ in range(words_per_doc):
        # Sample topic for this word
        topic_dist = tfd.Categorical(probs=theta[d])
        topic_idx = topic_dist.sample()

        # Sample word from chosen topic
        word_dist = tfd.Categorical(probs=topics[topic_idx])
        word_idx = word_dist.sample()

        words.append(vocab[word_idx.numpy()])

    print("Generated words:", " ".join(words))