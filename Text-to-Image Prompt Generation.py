from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

# Load T5 model
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

# User input
user_input = "Generate a prompt for a landscape painting."

# Format instruction for the model
input_text = "create a detailed image prompt: " + user_input

# Tokenize input
inputs = tokenizer(
    input_text,
    return_tensors="tf",
    max_length=128,
    truncation=True
)

# Generate prompt
outputs = model.generate(
    inputs["input_ids"],
    max_length=60,
    num_beams=5,
    temperature=0.9,
    early_stopping=True
)

# Decode output
generated_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Display results
print("📝 User Input:")
print(user_input)

print("\n🖼️ Generated Image Prompt:")
print(generated_prompt)