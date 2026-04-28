from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset
import numpy as np
import torch

# Sample dataset
data = {
    "text": [
        "The new iPhone looks amazing!",
        "The game was boring and unwatchable.",
        "I absolutely loved the movie!",
        "This laptop performs very poorly.",
        "Fantastic service and great food!"
    ],
    "label": [1, 0, 1, 0, 1]   # 1 = positive, 0 = negative
}

# Convert to HuggingFace dataset
dataset = Dataset.from_dict(data)
dataset = dataset.train_test_split(test_size=0.2)

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Tokenization function
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Tokenize dataset
tokenized_ds = dataset.map(tokenize, batched=True)

# Format for PyTorch
tokenized_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
)

# Train the model
trainer.train()

# ---------- Inference ----------
sample = "I hated the user interface of this app."

tokens = tokenizer(sample, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    output = model(**tokens)

prediction = torch.argmax(output.logits, dim=1).item()

print("📝 Sample:", sample)
print("🔖 Predicted Sentiment:", "Positive" if prediction == 1 else "Negative")