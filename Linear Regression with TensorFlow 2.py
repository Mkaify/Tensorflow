import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
 
# Generate synthetic training data
X = np.linspace(0, 10, 100)                           # 100 evenly spaced values from 0 to 10
y = 3 * X + 2 + np.random.randn(*X.shape) * 0.5       # Linear relation with noise
 
# Create a simple sequential model with one Dense unit
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])   # One neuron, input is 1D
])
 
# Compile the model with mean squared error loss and SGD optimizer
model.compile(optimizer='sgd', loss='mse')             # Optimizer: SGD, Loss: Mean Squared Error
 
# Train the model
model.fit(X, y, epochs=100, verbose=0)                 # Train for 100 epochs, silent output
 
# Predict values using the trained model
predictions = model.predict(X)                         # Predict for all X values
 
# Plot original data and predicted line
plt.scatter(X, y, label='Data')                        # Scatter plot of original data
plt.plot(X, predictions, color='red', label='Model')   # Red line: model predictions
plt.legend()
plt.title("Linear Regression with TensorFlow 2")
plt.show()