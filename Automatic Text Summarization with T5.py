from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

# Load pretrained T5 model and tokenizer
model_name = "t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

# Input text
text = """
Machine learning is a subset of artificial intelligence that provides systems the ability
to automatically learn and improve from experience without being explicitly programmed.
It focuses on the development of computer programs that can access data and use it to
learn for themselves.
"""

# Prepare prompt for T5
input_text = "summarize: " + text

# Modern tokenizer API
inputs = tokenizer(
    input_text,
    return_tensors="tf",
    max_length=512,
    truncation=True,
    padding=True
)

# Generate summary
outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=50,
    num_beams=4,
    early_stopping=True
)

# Decode summary
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Display results
print("📝 Original Text:\n", text.strip())
print("\n🔍 Generated Summary:\n", summary)