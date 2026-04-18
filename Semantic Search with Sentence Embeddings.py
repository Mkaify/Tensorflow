from sentence_transformers import SentenceTransformer, util

# Load pretrained SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Document corpus
documents = [
    "Machine learning models require a lot of data to perform well.",
    "The Eiffel Tower is located in Paris, France.",
    "Neural networks are a powerful tool in deep learning.",
    "I love visiting historical places during vacations.",
    "Transformers have revolutionized natural language processing."
]

# Encode documents into embeddings
doc_embeddings = model.encode(documents, convert_to_tensor=True)

# User query
query = "How do neural networks work in AI?"

# Encode query
query_embedding = model.encode(query, convert_to_tensor=True)

# Compute cosine similarities
cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

# Identify top matching document
top_result_idx = int(cos_scores.argmax())

# Display results
print("🔍 Query:\n", query)
print("\n📄 Top Matching Document:\n", documents[top_result_idx])
print("\n🔢 Similarity Score: {:.2f}".format(cos_scores[top_result_idx].item()))