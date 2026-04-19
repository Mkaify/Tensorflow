from sentence_transformers import SentenceTransformer, util
 
# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
 
# Define FAQ pairs (question → answer)
faq_questions = [
    "How do I reset my password?",
    "What is the refund policy?",
    "How can I contact customer support?",
    "Where can I find my order history?",
    "How do I update my account information?"
]
 
faq_answers = [
    "To reset your password, click 'Forgot Password' on the login screen.",
    "You can request a refund within 30 days of purchase.",
    "You can reach our support team via the Contact Us page.",
    "Go to your profile and click on 'Order History'.",
    "Visit Account Settings to update your personal info."
]
 
# Encode FAQ questions
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)
 
# User query
user_query = "I want to change the email linked to my account."
 
# Encode user query
query_embedding = model.encode(user_query, convert_to_tensor=True)
 
# Compute cosine similarities
cos_scores = util.pytorch_cos_sim(query_embedding, faq_embeddings)[0]
 
# Find best matching FAQ
best_match_idx = int(cos_scores.argmax())
print("❓ User Query:\n", user_query)
print("\n✅ Matched FAQ:\n", faq_questions[best_match_idx])
print("\n💬 Answer:\n", faq_answers[best_match_idx])
print("\n🔢 Similarity Score: {:.2f}".format(cos_scores[best_match_idx].item()))