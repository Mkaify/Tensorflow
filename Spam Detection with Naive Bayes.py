from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Sample dataset: messages and labels
messages = [
    "Congratulations! You've won a free iPhone. Click here to claim now!",
    "Reminder: your appointment is scheduled for tomorrow at 3 PM.",
    "Free entry in a weekly contest. Win $1,000 cash!",
    "Are we still meeting for lunch today?",
    "Get cheap meds online without a prescription.",
    "Don't forget to bring your laptop to the meeting."
]
labels = ["spam", "ham", "spam", "ham", "spam", "ham"]
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.33, random_state=42)
 
# Build pipeline: TF-IDF + Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
 
# Evaluate on test data
y_pred = model.predict(X_test)
print("📊 Spam Detection Report:\n")
print(classification_report(y_test, y_pred))
 
# Test on new message
new_msg = "Claim your prize now! You've been selected for a free vacation."
prediction = model.predict([new_msg])[0]
print("📨 New Message:\n", new_msg)
print("🔖 Predicted Label:", prediction)