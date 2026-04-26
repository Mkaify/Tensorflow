1from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# =========================
# Sample document corpus
# =========================
documents = [
    "Artificial intelligence and machine learning are transforming technology.",
    "Neural networks are widely used in deep learning models.",
    "The football team won the championship after a thrilling match.",
    "Basketball players require strength, speed, and coordination.",
    "AI applications include natural language processing and computer vision.",
    "The Olympic games feature the world's best athletes."
]

# =========================
# Convert text into TF-IDF features
# =========================
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# =========================
# Apply LDA for Topic Modeling
# =========================
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

# =========================
# Display Topics
# =========================
feature_names = vectorizer.get_feature_names_out()

print("📚 Discovered Topics:\n")
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-5:]]
    print(f"Topic {topic_idx+1}: {', '.join(top_words)}")

# =========================
# Assign topic to each document
# =========================
doc_topics = lda.transform(X)

print("\n📄 Document Topic Distribution:\n")
for i, doc in enumerate(documents):
    topic = doc_topics[i].argmax()
    print(f"Document: {doc}")
    print(f"→ Assigned Topic: {topic+1}\n")