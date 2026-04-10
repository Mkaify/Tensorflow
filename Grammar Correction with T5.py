from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

# Load grammar correction model
model_name = "vennify/t5-base-grammar-correction"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

# Sentence with grammar mistakes
sentence = "She no went to the store because it raining."

# Input format expected by model
input_text = "grammar: " + sentence + " </s>"

encoding = tokenizer.encode_plus(
    input_text,
    return_tensors="tf",
    max_length=128,
    truncation=True
)

input_ids = encoding["input_ids"]

# Generate corrected text
outputs = model.generate(
    input_ids,
    max_length=64,
    num_beams=5,
    early_stopping=True
)

corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("❌ Original:")
print(sentence)

print("\n✅ Corrected:")
print(corrected)