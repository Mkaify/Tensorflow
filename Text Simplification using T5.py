from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

# Load T5 model and tokenizer
model_name = "t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

# Complex sentence
text = "The precipitation will persist throughout the afternoon, primarily impacting regions with lower atmospheric pressure."

# Add task prefix
input_text = "simplify: " + text

input_ids = tokenizer.encode(
    input_text,
    return_tensors="tf",
    max_length=128,
    truncation=True
)

# Generate simplified text
outputs = model.generate(
    input_ids,
    max_length=50,
    num_beams=4,
    early_stopping=True
)

simplified = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("📚 Original Text:\n", text)
print("\n🧾 Simplified Version:\n", simplified)