from transformers import pipeline

# Load GPT-2 text generation pipeline
generator = pipeline("text-generation", model="gpt2", max_length=100)

# Define few-shot classification prompt
prompt = """
Classify the following text into categories: [greeting, booking, question, complaint]

Example 1:
Text: Hello, how are you?
Category: greeting

Example 2:
Text: I want to book a flight to Toronto.
Category: booking

Example 3:
Text: Can you tell me the train schedule?
Category: question

Example 4:
Text: My hotel room was dirty and not cleaned.
Category: complaint

Now classify:
Text: I need help canceling my reservation.
Category:"""

# Generate prediction deterministically
output = generator(prompt, do_sample=False)[0]['generated_text']

# Extract the predicted category (after 'Category:')
predicted_category = output.split("Category:")[-1].strip().split("\n")[0]

# Display result
print("📝 Prompted Classification Result:\n", predicted_category)