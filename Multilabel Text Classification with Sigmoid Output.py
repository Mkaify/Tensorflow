from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Sample document dataset
documents = [
    "Artificial intelligence and machine learning are advancing rapidly.",
    "Neural networks power many deep learning applications.",
    "The football team won the national championship.",
    "Basketball players train hard to improve performance.",
    "New smartphones feature advanced processors and cameras.",
    "Technology companies are investing heavily in AI research."
]

# Convert documents to TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(documents)

# Apply K-Means clustering
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Display cluster assignments
print("📚 Document Clusters:\n")
for i, doc in enumerate(documents):
    print(f"Document: {doc}")
    print(f"→ Cluster: {kmeans.labels_[i]}\n")

# Show top words in each cluster
terms = vectorizer.get_feature_names_out()
print("🔑 Top Terms per Cluster:\n")

for i in range(num_clusters):
    center_terms = kmeans.cluster_centers_[i].argsort()[-5:]
    top_words = [terms[j] for j in center_terms]
    print(f"Cluster {i}: {', '.join(top_words)}")