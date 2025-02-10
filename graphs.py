# graphs.py
import numpy as np
import matplotlib.pyplot as plt
from lstm import train_lstm_model

def create_sequences(data, time_step):
    """
    Creates sequences of data suitable for LSTM input.
    
    Parameters:
        data (np.array): The data to create sequences from.
        time_step (int): The number of time steps to include in each sequence.
    
    Returns:
        X (np.array): The input data reshaped into sequences.
        y (np.array): The target values (what we're trying to predict).
    """
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])  # Input sequence
        y.append(data[i+time_step])    # Target value (next value in the sequence)
    return np.array(X), np.array(y)

# Example synthetic data (e.g., stock prices or any time series)
data = np.sin(np.linspace(0, 100, 1000))  # Example sine wave data as a placeholder

# Define the time step for LSTM sequences
time_step = 50  # You can adjust this based on your data's time granularity

# Create sequences
X, y = create_sequences(data, time_step)

# Reshape X for LSTM: [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train the LSTM model and get history
model, history, X_test, y_test = train_lstm_model(X, y)

# Function to plot training history
def plot_training_history(history):
    """
    Plots training and validation loss over epochs.
    
    Parameters:
        history (History): The history object from model training.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Function to plot predictions vs true values
def plot_predictions(model, X_test, y_test):
    """
    Plots predictions against true values.
    
    Parameters:
        model (tf.keras.Model): The trained model.
        X_test (np.array): The test feature set.
        y_test (np.array): The true target values for the test set.
    """
    predictions = model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True Values')
    plt.plot(predictions, label='Predictions')
    plt.title('Predictions vs True Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Plot training history
plot_training_history(history)

# Plot predictions
plot_predictions(model, X_test, y_test)
