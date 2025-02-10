import numpy as np
import pandas as pd
from scipy import signal
import plotly.graph_objects as go
import os

# Load the dataset (replace with the actual path to your dataset file)
eeg_data = pd.read_csv(r"C:\Users\david\Downloads\data.csv")

# Drop unnecessary index column if it exists
eeg_data = eeg_data.drop(columns=['Unnamed: 0'])

# Inspect columns to ensure correct column name
print("Columns in the dataset:", eeg_data.columns)

# Select the first 5 rows, excluding the first and last columns
eeg_data = eeg_data.iloc[:5, 1:-1]  # First 5 rows, excluding the first column and the last column 'y'

# Create a directory to save the plots
output_dir = "eeg_spectrograms"
os.makedirs(output_dir, exist_ok=True)

# Sampling frequency in Hz (adjust according to your data)
fs = 1000  # Modify this value to match your data's sampling frequency

# Log message to start processing
print(f"Processing {len(eeg_data)} rows of EEG data...")

# Loop through each row to create a spectrogram for each
for i, row in eeg_data.iterrows():
    # Log progress for each row
    print(f"Processing Row {i+1}/{len(eeg_data)}")

    # Get the EEG signal for this row (all columns except 'y' and the first column)
    eeg_signal = row.values  # This gives the values for each row (excluding the first and last column)

    # Check for missing values
    if np.any(np.isnan(eeg_signal)):
        print(f"Warning: Missing values detected in row {i}. Removing NaNs.")
        eeg_signal = eeg_signal[~np.isnan(eeg_signal)]  # Remove NaNs

    # Check if the length of the signal is less than 178 (for nperseg), warn if so
    if len(eeg_signal) < 178:
        print(f"Warning: Signal length in Row {i} is less than 178 samples. Adjusting nperseg.")

    # Generate the spectrogram with custom nperseg
    nperseg_value = min(len(eeg_signal), 178)  # Use the length of the signal or 178, whichever is smaller
    try:
        frequencies, times, Sxx = signal.spectrogram(eeg_signal, fs, nperseg=nperseg_value)
    except Exception as e:
        print(f"Error generating spectrogram for Row {i}: {e}")
        continue  # Skip this row if there's an error

    # Ensure the power values are not zero (avoid log(0) issue)
    Sxx = np.log(Sxx + 1e-10)

    # Create a Plotly figure for the spectrogram
    try:
        fig = go.Figure(data=go.Heatmap(
            z=Sxx, 
            x=times, 
            y=frequencies, 
            colorscale='Jet', 
            colorbar=dict(title='Log Power')
        ))

        # Update layout for better labeling
        fig.update_layout(
            title=f"EEG Spectrogram for Row {i}",
            xaxis_title="Time [sec]",
            yaxis_title="Frequency [Hz]",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            height=500,
            width=800
        )

        # Save the plot as an image (e.g., PNG format)
        image_path = os.path.join(output_dir, f"eeg_spectrogram_row_{i}.png")
        fig.write_image(image_path)

        # Log message to confirm saving each image
        print(f"Spectrogram for Row {i} saved at {image_path}")

    except Exception as e:
        print(f"Error generating Plotly figure for Row {i}: {e}")
        continue  # Skip this row if there's an error

# Verify if the images were saved successfully
saved_files = os.listdir(output_dir)
print(f"Files in '{output_dir}': {saved_files}")

print("Spectrograms saved to folder:", output_dir)
