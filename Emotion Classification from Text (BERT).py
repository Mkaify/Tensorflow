import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np

# ===============================
# 0️⃣ Clean Logs + Reproducibility
# ===============================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.keras.utils.set_random_seed(42)

# ===============================
# 1️⃣ Load Pretrained BERT
# ===============================
bert_preprocess = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    name="preprocessing"
)

bert_encoder = hub.KerasLayer(
    "https://tfhub.dev/google/experts/bert/wiki_books/sst2/2",
    trainable=True,   # 🔥 Fine-tune BERT
    name="BERT_encoder"
)

# ===============================
# 2️⃣ Example Dataset
# ===============================
texts = [
    "I am so happy today!",
    "This is absolutely terrible",
    "I'm feeling really down",
    "You did a great job!",
    "Why would you say that?"
]

labels = np.array([0, 1, 2, 0, 1])  # 0=joy, 1=anger, 2=sadness

# ===============================
# 3️⃣ Train/Validation Split
# ===============================
split = int(0.8 * len(texts))
train_texts, val_texts = texts[:split], texts[split:]
train_labels, val_labels = labels[:split], labels[split:]

# ===============================
# 4️⃣ tf.data Pipeline
# ===============================
batch_size = 2

train_ds = tf.data.Dataset.from_tensor_slices((train_texts, train_labels)) \
    .shuffle(10) \
    .batch(batch_size) \
    .prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_texts, val_labels)) \
    .batch(batch_size) \
    .prefetch(tf.data.AUTOTUNE)

# ===============================
# 5️⃣ Build End-to-End Model
# ===============================
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")

encoder_inputs = bert_preprocess(text_input)
encoder_outputs = bert_encoder(encoder_inputs)

pooled_output = encoder_outputs["pooled_output"]

x = tf.keras.layers.Dense(64, activation="relu")(pooled_output)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(3, activation="softmax")(x)

model = tf.keras.Model(text_input, output)

model.summary()

# ===============================
# 6️⃣ Compile
# ===============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),  # 🔥 small LR for BERT
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# ===============================
# 7️⃣ Train
# ===============================
model.fit(train_ds, validation_data=val_ds, epochs=5)

# ===============================
# 8️⃣ Inference Function
# ===============================
def classify_emotion(text):
    pred = model.predict([text])[0]
    emotion = ["Joy", "Anger", "Sadness"][np.argmax(pred)]
    return emotion

print("Text: I can’t believe how amazing this is!")
print("Predicted Emotion:", classify_emotion(
    "I can’t believe how amazing this is!"
))