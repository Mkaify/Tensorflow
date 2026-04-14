import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import numpy as np
 
# Sample intents dataset
sentences = [
    "I want to book a flight to New York",   # booking
    "Hey there!",                            # greeting
    "What’s the weather like today?",        # question
    "Can you help me cancel my ticket?",     # cancel
    "Good morning, how are you?"             # greeting
]
labels = [0, 1, 2, 3, 1]  # intent labels: 0=booking, 1=greeting, 2=question, 3=cancel
 
# Load BERT base and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert = TFAutoModel.from_pretrained(model_name)
 
# Tokenize inputs
tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors='tf')
 
# Build intent classification model
input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
 
bert_outputs = bert(input_ids, attention_mask=attention_mask)[1]  # [1] = pooled output
x = tf.keras.layers.Dense(64, activation='relu')(bert_outputs)
output = tf.keras.layers.Dense(4, activation='softmax')(x)  # 4 intent classes
 
model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
# Train the model
model.fit(
    {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']},
    np.array(labels),
    epochs=5,
    verbose=0
)
 
# Inference
test_input = "I'd like to cancel my reservation"
test_tokens = tokenizer(test_input, return_tensors='tf', truncation=True, padding=True)
pred = model.predict({'input_ids': test_tokens['input_ids'], 'attention_mask': test_tokens['attention_mask']})
intent = ["Booking", "Greeting", "Question", "Cancel"][np.argmax(pred)]
print("🧑 Input:", test_input)
print("📌 Predicted Intent:", intent)