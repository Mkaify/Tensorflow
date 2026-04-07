from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

# Load pretrained T5 model and tokenizer
model_name = "t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

# Input context
context = "Albert Einstein was a physicist who developed the theory of relativity."

# Format prompt for T5
input_text = "generate question: " + context

# Tokenize input (recommended modern API)
inputs = tokenizer(
    input_text,
    return_tensors="tf",
    padding=True,
    truncation=True
)

# Generate question
outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=32,
    num_beams=4,
    early_stopping=True
)

# Decode output
question = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("📄 Context:\n", context)
print("\n❓Generated Question:\n", question)