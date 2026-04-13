from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load DialoGPT model
model_name = "microsoft/DialoGPT-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

chat_history_ids = None

# User input
user_input = "Hi there! How are you today?"

# Encode input
new_input_ids = tokenizer.encode(
    user_input + tokenizer.eos_token,
    return_tensors='pt'
)

# Append to chat history
bot_input_ids = (
    torch.cat([chat_history_ids, new_input_ids], dim=-1)
    if chat_history_ids is not None
    else new_input_ids
)

# Generate response
chat_history_ids = model.generate(
    bot_input_ids,
    max_length=1000,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8
)

# Decode response
response = tokenizer.decode(
    chat_history_ids[:, bot_input_ids.shape[-1]:][0],
    skip_special_tokens=True
)

print("🧑 You:", user_input)
print("🤖 Bot:", response)