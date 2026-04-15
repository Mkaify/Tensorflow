from transformers import pipeline
 
# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
 
# Input sentence and candidate labels
text = "I need to book a hotel for my trip next week."
candidate_labels = ["travel", "finance", "education", "greeting"]
 
# Perform zero-shot classification
result = classifier(text, candidate_labels)
 
# Display results
print("🧾 Text:\n", text)
print("\n🏷️ Zero-shot Classification Results:")
for label, score in zip(result['labels'], result['scores']):
    print(f"  {label}: {score:.2f}")