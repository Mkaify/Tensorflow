from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =========================
# Sample Dataset
# =========================
texts = [
    "Artificial intelligence is transforming the tech industry.",
    "The new iPhone has amazing camera capabilities.",
    "Manchester United won their last football match.",
    "The Lakers are a strong basketball team.",
    "Quantum computing will shape the future of science.",
    "The Olympics host the best athletes in the world."
]

labels = ["Tech", "Tech", "Sports", "Sports", "Tech", "Sports"]

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.33, random_state=42
)

# =========================
# Build pipeline: TF-IDF + Linear SVM
# =========================
model = make_pipeline(
    TfidfVectorizer(stop_words='english', max_features=5000),
    LinearSVC()
)

# Train the classifier
model.fit(X_train, y_train)

# =========================
# Evaluate the model
# =========================
y_pred = model.predict(X_test)
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# Predict new document
# =========================
new_doc = "Basketball players need excellent stamina and coordination."
predicted_label = model.predict([new_doc])[0]

print("📝 New Document:\n", new_doc)
print("🔖 Predicted Category:", predicted_label)