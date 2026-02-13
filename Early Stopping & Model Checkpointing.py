import tensorflow as tf
import numpy as np
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ================================
# Generate Synthetic Data
# ================================
X = np.linspace(0, 10, 300).reshape(-1, 1).astype(np.float32)
y = 2 * X + 3 + np.random.normal(0, 0.5, size=X.shape)


# ================================
# Create Model
# ================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),                 # Proper Input Layer
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


# ================================
# Compile
# ================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)


# ================================
# Callbacks
# ================================

# Early Stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Model Checkpoint (Modern Format)
checkpoint_path = "best_model.keras"

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=0
)


# ================================
# Train
# ================================
history = model.fit(
    X, y,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=0
)


# ================================
# Load Best Model (Optional Safety)
# ================================
best_model = tf.keras.models.load_model(checkpoint_path)


# ================================
# Evaluate
# ================================
loss, mae = best_model.evaluate(X, y, verbose=0)

print(f"Final MAE: {mae:.3f}")
