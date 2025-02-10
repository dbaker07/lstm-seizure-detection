import os
import zipfile
import numpy as np
import wfdb
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow detected {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f" - {gpu}")
else:
    print("No GPU detected. TensorFlow is running on CPU.")

# --- Step 1: Extract ZIP File ---
zip_path = r"C:\Users\david\Downloads\post-ictal-heart-rate-oscillations-in-partial-epilepsy-1.0.0.zip"
dataset_path = r"C:\Users\david\Downloads\szdb_extracted\post-ictal-heart-rate-oscillations-in-partial-epilepsy-1.0.0"

if not os.path.exists(dataset_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
    print("Dataset extracted successfully.")

# --- Step 2: Load ECG Data ---
def load_ecg_data(record_name):
    record_path = os.path.join(dataset_path, record_name)
    try:
        ecg_record = wfdb.rdsamp(record_path)
        annotation = wfdb.rdann(record_path, 'ari')
        ecg_data = np.array(ecg_record[0])
        seizure_labels = np.zeros(len(ecg_data))
        for i in annotation.sample:
            seizure_labels[i] = 1
        return ecg_data, seizure_labels
    except Exception as e:
        print(f"Error loading {record_name}: {e}")
        return None, None

ecgs, labels = [], []
for file in os.listdir(dataset_path):
    if file.endswith(".hea"):
        record_name = file[:-4]
        ecg, label = load_ecg_data(record_name)
        if ecg is not None:
            ecgs.append(ecg)
            labels.append(label)

if ecgs:
    ecg_data = np.vstack(ecgs)
    labels = np.hstack(labels)
else:
    raise ValueError("No valid ECG data found!")

# --- Step 3: Preprocess the Data ---
time_steps = 100
scaler = StandardScaler()
ecgs_scaled = scaler.fit_transform(ecg_data).astype(np.float32)
train_ecgs, val_ecgs, train_labels, val_labels = train_test_split(
    ecgs_scaled, labels, test_size=0.2, random_state=42, stratify=labels
)

# --- Step 4: Data Generator ---
def data_generator(batch_size, ecgs_scaled, labels, time_steps):
    num_samples = len(ecgs_scaled)
    while True:
        for i in range(0, num_samples - time_steps, batch_size):
            X_batch, y_batch = [], []
            for j in range(i, min(i + batch_size, num_samples - time_steps)):
                X_batch.append(ecgs_scaled[j: j + time_steps])
                y_batch.append(labels[j + time_steps])
            yield np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)

# --- Step 5: Define the CNN Model ---
def build_ecg_seizure_detection_model(input_shape=(100, 1)):
    model = models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(256, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.7),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model_filename = 'ecg_seizure_detection_model.h5'
if os.path.exists(model_filename):
    model = load_model(model_filename)
    print("Model loaded from file.")
else:
    model = build_ecg_seizure_detection_model()
    print("New model created.")

# --- Step 6: Class Weight Calculation ---
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall', 'AUC']
)

# --- Step 7: Callbacks ---
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)

# --- Step 8: Train Model ---
batch_size = 64
steps_per_epoch = (len(train_ecgs) - time_steps) // batch_size
validation_steps = (len(val_ecgs) - time_steps) // batch_size

history = model.fit(
    data_generator(batch_size, train_ecgs, train_labels, time_steps),
    steps_per_epoch=steps_per_epoch,
    validation_data=data_generator(batch_size, val_ecgs, val_labels, time_steps),
    validation_steps=validation_steps,
    epochs=25,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr]
)

model.save(model_filename)

# --- Step 9: Evaluation Fix ---
test_batch_size = 64
test_steps = (len(ecgs_scaled) - time_steps) // test_batch_size

eval_results = model.evaluate(
    data_generator(test_batch_size, ecgs_scaled, labels, time_steps),
    steps=test_steps
)
metrics_names = model.metrics_names
print("Test Results:", dict(zip(metrics_names, eval_results)))

# --- Step 10: Predictions ---
y_pred = (model.predict(
    data_generator(test_batch_size, ecgs_scaled, labels, time_steps),
    steps=test_steps
) > 0.5).astype("int32")

cm = confusion_matrix(labels[:len(y_pred)], y_pred)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(labels[:len(y_pred)], y_pred))