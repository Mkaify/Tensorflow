from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# Model and tokenizer
model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Example document
document = """
Artificial Intelligence (AI) is transforming the world around us. From voice assistants and self-driving cars
to medical diagnostics and financial predictions, AI systems are now integral to modern life. At its core, AI
involves creating machines that can mimic human intelligence and improve themselves through data-driven learning.
"""

# Tokenize input
inputs = tokenizer(document, truncation=True, padding="longest", return_tensors="pt")

# Generate summary
summary_ids = model.generate(**inputs)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Original Document:\n", document.strip())
print("\nGenerated Summary:\n", summary)