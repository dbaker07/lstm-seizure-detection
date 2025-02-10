import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras import mixed_precision
import seaborn as sns
import matplotlib.pyplot as plt

# Enable mixed precision for GPU optimization
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Load the EEG dataset from the CSV file (replace with the correct path)
eeg_data = pd.read_csv(r"C:\Users\david\Downloads\data.csv")  # Replace with the path to your CSV file

# Check the first few rows and columns
print("Columns in the dataset:", eeg_data.columns)

# Access the EEG signals (X) and labels (y)
X = eeg_data.iloc[:, 1:-1].values  # EEG signals (exclude first column and last column)
y = eeg_data.iloc[:, -1].values  # Labels (last column)

# Inspect the shapes to verify
print("EEG signals shape:", X.shape)
print("Labels shape:", y.shape)

# Modify the labels as requested
y = np.where(y > 1, 1, 0)  # Convert labels 2, 3, 4, 5 to 1, and 1 to 0

# Inspect the modified labels
print("Modified Labels:", np.unique(y))

# Normalize EEG data (scaling values between 0 and 1)
X = X / X.max(axis=0)

# Reshape EEG data for RNN (LSTM) input - Add a new dimension for time_steps = 1
X = X.reshape((X.shape[0], 1, X.shape[1]))  # (samples, time_steps, features)

# Set up K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameters
learning_rate = 1e-4
batch_size = 64  # Start with 64, can be increased based on available memory
epochs = 20  # Set to 20 initially, adjust if needed
accuracies = []
f1_scores = []

# Cross-validation loop
for train_index, val_index in kf.split(X):
    # Split data into training and validation sets
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Build the model for each fold
    def build_seizure_prediction_rnn(input_shape=(1, 178), learning_rate=1e-4):
        model = models.Sequential()
        model.add(layers.LSTM(64, activation='tanh', input_shape=input_shape, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(32, activation='tanh'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='sigmoid'))  # Binary output: 0 for no seizure, 1 for seizure
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Create model
    model = build_seizure_prediction_rnn(input_shape=(1, X.shape[2]), learning_rate=learning_rate)
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model_fold.h5', monitor='val_loss', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
    
    # Train the model with the current fold's data
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                        callbacks=[early_stopping, model_checkpoint, reduce_lr], verbose=1)
    
    # Evaluate the model on the validation set
    val_accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
    accuracies.append(val_accuracy)
    
    # Make predictions on the validation set
    y_pred_probs = model.predict(X_val)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Calculate F1 score for the fold
    f1 = f1_score(y_val, y_pred)
    f1_scores.append(f1)

    # Confusion Matrix for each fold
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Seizure', 'Seizure'], yticklabels=['No Seizure', 'Seizure'])
    plt.title(f'Confusion Matrix for Fold {len(accuracies)}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Calculate average accuracy and F1 score across all folds
avg_accuracy = np.mean(accuracies)
avg_f1_score = np.mean(f1_scores)

print(f'Average Accuracy across folds: {avg_accuracy * 100:.2f}%')
print(f'Average F1 Score across folds: {avg_f1_score:.4f}')

# After cross-validation, you can train the model on the full dataset or best fold

# Evaluate on the test set using the best model from cross-validation
best_model = models.load_model('best_model_fold.h5')

# Evaluate the model
test_loss, test_accuracy = best_model.evaluate(X_val, y_val)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
best_model.save('final_seizure_model.h5')
print("Final Model saved to final_seizure_model.h5")

# Make predictions on the test set
y_pred_probs = best_model.predict(X_val)  # Get predicted probabilities (0 to 1)

# Convert probabilities to binary predictions (0 or 1)
y_pred = (y_pred_probs > 0.5).astype(int)

# Calculate the F1 score
f1 = f1_score(y_val, y_pred)
print(f"Final F1 Score: {f1:.4f}")

# Confusion Matrix for the final model on the validation set
final_cm = confusion_matrix(y_val, y_pred)
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Seizure', 'Seizure'], yticklabels=['No Seizure', 'Seizure'])
plt.title('Final Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Making predictions on a new EEG signal (example on the first validation sample)
new_eeg_signal = np.expand_dims(X_val[0], axis=0)  # Add batch dimension
prediction = best_model.predict(new_eeg_signal)
print("Prediction (0: no seizure, 1: seizure):", prediction[0][0])
