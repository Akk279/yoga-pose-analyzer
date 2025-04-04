import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sample Model (You can skip this if your model is already trained)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),  # Example input shape
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Dummy Training Data (Replace with real dataset)
import numpy as np
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=8)
