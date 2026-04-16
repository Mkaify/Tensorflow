from sentence_transformers import SentenceTransformer, util

# Load pretrained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample sentences
sentences = [
    "I love machine learning.",
    "Deep learning is a branch of AI.",
    "Let's grab a cup of coffee.",
    "Artificial intelligence is transforming the world."
]

# Generate embeddings (as PyTorch tensors)
embeddings = model.encode(sentences, convert_to_tensor=True)

# Compute cosine similarity matrix
cos_sim = util.pytorch_cos_sim(embeddings, embeddings)

# Display similarity matrix
print("🔢 Cosine Similarity Matrix:")
for i, s in enumerate(sentences):
    similarities = ["{:.2f}".format(score) for score in cos_sim[i]]
    print(f"{s}\n → {similarities}\n")