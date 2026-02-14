import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ================================
# Set Random Seeds
# ================================
np.random.seed(42)
tf.random.set_seed(42)


# ================================
# Load Dataset
# ================================
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv(url, names=columns)


# ================================
# Features & Target
# ================================
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values


# ================================
# Feature Scaling
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ================================
# Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y        # Maintain class balance
)


# ================================
# Build Model
# ================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X.shape[1],)),     # Proper Input Layer
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# ================================
# Compile Model
# ================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# ================================
# Callbacks (Early Stopping)
# ================================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)


# ================================
# Train Model
# ================================
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=0
)


# ================================
# Evaluate Model
# ================================
loss, acc = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Accuracy: {acc:.3f}")
