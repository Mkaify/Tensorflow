from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

# Load pretrained T5 model
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

# Example table row
table_row = {
    "Name": "Alice Johnson",
    "Age": "29",
    "Occupation": "Software Engineer",
    "Location": "San Francisco"
}

# Convert table row into text format
table_text = " | ".join([f"{k}: {v}" for k, v in table_row.items()])

# T5 prompt
input_text = "generate description: " + table_text

# Tokenize
inputs = tokenizer(
    input_text,
    return_tensors="tf",
    max_length=128,
    truncation=True
)

# Generate text
outputs = model.generate(
    inputs["input_ids"],
    max_length=60,
    num_beams=4,
    early_stopping=True
)

# Decode output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print results
print("📊 Table Row:\n", table_row)
print("\n📝 Generated Description:\n", generated_text)