import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense

# Generate synthetic data

data = np.random.normal(0, 1, (1000, 20)) # Normal data

anomalies = np.random.normal(0, 5, (50, 20)) # Anomalous data

# Combine normal and anomalous data

dataset = np.concatenate([data, anomalies], axis=0)

labels = np.concatenate([np.zeros(1000), np.ones(50)], axis=0) # 0 for normal, 1 for anomaly

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# Standardize the data

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Define the size of the input

input_dim = X_train.shape[1]

encoding_dim = 10 # Latent space dimension

# Encoder

input_layer = Input(shape=(input_dim,))

encoded = Dense(encoding_dim, activation='relu')(input_layer)

# Decoder

decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder model

autoencoder = Model(inputs=input_layer, outputs=decoded)

# Compile the autoencoder model

autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder

history = autoencoder.fit(X_train, X_train,

 epochs=50,

 batch_size=32,

 validation_data=(X_test, X_test),

 shuffle=True)

# Predict the reconstructed data

X_test_pred = autoencoder.predict(X_test)

# Calculate the reconstruction error

mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

# Define a threshold for anomaly detection

threshold = np.percentile(mse, 95) # Choose a threshold based on 95th percentile

# Identify anomalies

anomalies = mse > threshold

# Print results

print(f"Detected {np.sum(anomalies)} anomalies out of {len(X_test)} samples.")

# Plot the training and validation loss

plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend()

plt.show()

# 1. Plot the distribution of reconstruction error

plt.figure(figsize=(10, 5))

plt.hist(mse, bins=50, alpha=0.6, color='b', label='Reconstruction Error')

plt.axvline(threshold, color='r', linestyle='--', label='Threshold')

plt.title('Distribution of Reconstruction Error')

plt.xlabel('Mean Squared Error')

plt.ylabel('Frequency')

plt.legend()

plt.show()

# 2. Plot a few examples of normal and anomalous data

# Selecting a few indices for normal and anomalous data

normal_indices = np.where(y_test == 0)[0][:5]

anomalous_indices = np.where(y_test == 1)[0][:5]

plt.figure(figsize=(15, 5))

for i, idx in enumerate(normal_indices):

plt.subplot(2, 5, i + 1)

 plt.plot(X_test[idx], label='Original', color='blue')

 plt.plot(X_test_pred[idx], label='Reconstructed', color='orange')

 plt.title('Normal Sample')

 plt.legend()

for i, idx in enumerate(anomalous_indices):

 plt.subplot(2, 5, i + 6)

 plt.plot(X_test[idx], label='Original', color='blue')

 plt.plot(X_test_pred[idx], label='Reconstructed', color='orange')

 plt.title('Anomalous Sample')

 plt.legend()

plt.tight_layout()

plt.show()

# 3. Plot the reconstruction error for individual samples

plt.figure(figsize=(10, 5))

plt.plot(mse, label='Reconstruction Error', color='b')

plt.axhline(threshold, color='r', linestyle='--', label='Threshold')

plt.title('Reconstruction Error per Sample')

plt.xlabel('Sample Index')

plt.ylabel('Reconstruction Error (MSE)')

plt.legend()

plt.show()
