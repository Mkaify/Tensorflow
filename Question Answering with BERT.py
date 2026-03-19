import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Required

import numpy as np

# -----------------------------------
# Load Official BERT QA model
# -----------------------------------
qa_model = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12_qa/3")

# Load matching preprocessor
preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

# -----------------------------------
# Context and Question
# -----------------------------------
context = """
TensorFlow is an end-to-end open-source platform for machine learning.
It has a comprehensive, flexible ecosystem of tools, libraries and community resources
that lets researchers innovate with machine learning and productionize AI easily.
"""

question = "What is TensorFlow used for?"

# -----------------------------------
# Preprocess
# -----------------------------------
inputs = preprocessor(
    text=[question],
    text_pair=[context]
)

# -----------------------------------
# Run Model
# -----------------------------------
outputs = qa_model(inputs)

start_logits = outputs["start_logits"][0]
end_logits = outputs["end_logits"][0]

# Get most probable start and end
start_index = tf.argmax(start_logits).numpy()
end_index = tf.argmax(end_logits).numpy()

# -----------------------------------
# Convert tokens back to words
# -----------------------------------
input_word_ids = inputs["input_word_ids"][0].numpy()

# Load vocab from preprocessor
vocab_file = preprocessor.resolved_object.vocab_file.asset_path.numpy()
tokenizer = tf.keras.layers.TextVectorization()
vocab = tf.io.gfile.GFile(vocab_file).read().splitlines()

tokens = [vocab[i] for i in input_word_ids]

# Extract answer tokens
answer_tokens = tokens[start_index:end_index + 1]

# Clean WordPiece tokens
answer = " ".join(answer_tokens).replace(" ##", "")

print("Question:", question)
print("Answer:", answer)