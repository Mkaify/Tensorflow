import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# Load sample dataset (Jena climate dataset from TensorFlow)
# url = "https://storage.googleapis.com/download.tensorflow.org/data/jena_climate_2009_2016.csv"
# path = tf.keras.utils.get_file("jena_climate.csv", origin=url)
df = pd.read_csv("datasets/jena_climate_2009_2016.csv")
 
# Use temperature column (in Celsius) for univariate forecasting
temps = df["T (degC)"].values.astype(np.float32)
 
# Normalize data
mean = temps.mean()
std = temps.std()
temps = (temps - mean) / std
 
# Create input-output pairs (sequence of 24 → next value)
def create_dataset(series, input_len=24, pred_len=1):
    X, y = [], []
    for i in range(len(series) - input_len - pred_len):
        X.append(series[i:i+input_len])
        y.append(series[i+input_len:i+input_len+pred_len])
    return np.array(X), np.array(y)
 
X, y = create_dataset(temps, input_len=24)
X = X[..., np.newaxis]  # Add channel dimension
 
# Split into train and test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
 
# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(24, 1)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1)
])
 
# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=5, validation_split=0.2)
 
# Predict on test data
preds = model.predict(X_test[:100]).flatten()
true = y_test[:100].flatten()
 
# Plot predictions
plt.plot(true, label="True")
plt.plot(preds, label="Predicted")
plt.title("Temperature Forecasting (Next Hour)")
plt.xlabel("Hour")
plt.ylabel("Normalized Temp")
plt.legend()
plt.show()