from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Sample document corpus
documents = [
    "Artificial intelligence and machine learning are transforming technology.",
    "Deep learning models require large amounts of data.",
    "The football team won the championship game.",
    "Basketball players need excellent teamwork and stamina.",
    "New smartphones feature advanced processors and cameras.",
    "Tennis tournaments attract players from around the world."
]

# Convert text to document-term matrix
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Train LDA model
lda = LatentDirichletAllocation(
    n_components=2,       # number of topics
    random_state=42
)

lda.fit(X)

# Get vocabulary
words = vectorizer.get_feature_names_out()

# Display top words per topic
print("📊 Discovered Topics:\n")

for topic_idx, topic in enumerate(lda.components_):
    top_words = [words[i] for i in topic.argsort()[-6:]]
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

# Test on a new document
new_doc = ["The latest AI models are improving rapidly."]
new_X = vectorizer.transform(new_doc)

topic_distribution = lda.transform(new_X)

print("\n📝 New Document:", new_doc[0])
print("📌 Topic Distribution:", topic_distribution)