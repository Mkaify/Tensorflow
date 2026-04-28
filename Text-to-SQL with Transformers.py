from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Smaller and easier model for text-to-SQL
model_name = "tscholak/cxmefzzi"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Natural language question
question = "List all customers who made a purchase in 2023."

# Database schema
schema = "Table customers(customer_id, name, email), orders(order_id, customer_id, date)"

# Format model input
input_text = f"translate English to SQL: {schema} | {question}"

# Tokenize input
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    max_length=256,
    truncation=True
)

# Generate SQL
outputs = model.generate(
    **inputs,
    max_length=128,
    num_beams=4,
    early_stopping=True
)

# Decode result
sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Display result
print("❓ Question:\n", question)
print("\n🧱 Schema:\n", schema)
print("\n📝 Generated SQL:\n", sql_query)