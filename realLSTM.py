import os
import numpy as np
import wfdb
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryFocalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow detected {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f" - {gpu}")
else:
    print("No GPU detected. TensorFlow is running on CPU.")

# --- Step 1: Load ECG Data ---
dataset_path = r"C:\Users\david\Downloads\szdb_extracted\post-ictal-heart-rate-oscillations-in-partial-epilepsy-1.0.0"

def load_ecg_data(record_name):
    record_path = os.path.join(dataset_path, record_name)
    
    try:
        ecg_record = wfdb.rdsamp(record_path)
        ecg_data = np.array(ecg_record[0])
        
        annotation_suffixes = ['ari', 'atr', 'qrs', 'st']
        annotation = None
        
        for suffix in annotation_suffixes:
            try:
                annotation = wfdb.rdann(record_path, suffix)
                break
            except:
                continue
        
        if annotation is None:
            print(f"Warning: No valid annotations found for {record_name}")
            return None, None
        
        seizure_labels = np.zeros(len(ecg_data), dtype=np.int32)
        seizure_labels[annotation.sample] = 1
        return ecg_data, seizure_labels
    
    except Exception as e:
        print(f"Error loading {record_name}: {e}")
        return None, None

record_names = [file[:-4] for file in os.listdir(dataset_path) if file.endswith(".hea")]
ecgs, labels = [], []

for record in record_names:
    ecg, label = load_ecg_data(record)
    if ecg is not None:
        ecgs.append(ecg)
        labels.append(label)

if not ecgs:
    raise ValueError("No valid ECG records found. Check dataset path or file format.")

ecg_data = np.vstack(ecgs)
labels = np.hstack(labels)

# --- Step 2: Handle Class Imbalance ---
seizure_indices = np.where(labels == 1)[0]
non_seizure_indices = np.where(labels == 0)[0]
if len(seizure_indices) > 0:
    seizure_upsampled = resample(seizure_indices, replace=True, n_samples=len(non_seizure_indices), random_state=42)
    balanced_indices = np.concatenate([non_seizure_indices, seizure_upsampled])
else:
    balanced_indices = non_seizure_indices
np.random.shuffle(balanced_indices)

ecg_data_balanced = ecg_data[balanced_indices]
labels_balanced = labels[balanced_indices]

# --- Step 3: Preprocess the Data ---
scaler = StandardScaler()
ecg_scaled = scaler.fit_transform(ecg_data_balanced).astype(np.float32)
time_steps = 100

if len(ecg_scaled) <= time_steps:
    raise ValueError("Dataset too small for selected time_steps!")

train_ecgs, temp_ecgs, train_labels, temp_labels = train_test_split(
    ecg_scaled, labels_balanced, test_size=0.25, random_state=42, stratify=labels_balanced
)
val_ecgs, test_ecgs, val_labels, test_labels = train_test_split(
    temp_ecgs, temp_labels, test_size=0.4, random_state=42, stratify=temp_labels
)

def data_generator(batch_size, ecgs_scaled, labels, time_steps):
    num_samples = len(ecgs_scaled)
    while True:
        indices = np.random.randint(0, num_samples - time_steps, batch_size)
        X_batch = np.array([ecgs_scaled[i:i+time_steps] for i in indices], dtype=np.float32).reshape(batch_size, time_steps, 1)
        y_batch = np.array([labels[i + time_steps - 1] for i in indices], dtype=np.float32)
        yield X_batch, y_batch

batch_size = 256
steps_per_epoch = len(train_ecgs) // batch_size
validation_steps = len(val_ecgs) // batch_size
test_steps = len(test_ecgs) // batch_size

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(batch_size, train_ecgs, train_labels, time_steps),
    output_signature=(
        tf.TensorSpec(shape=(None, time_steps, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
).shuffle(1000).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(batch_size, val_ecgs, val_labels, time_steps),
    output_signature=(
        tf.TensorSpec(shape=(None, time_steps, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
).shuffle(500).prefetch(tf.data.AUTOTUNE)

def build_lstm_model(input_shape=(100, 1)):
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.LSTM(128),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model_filename = r"C:\Users\david\vs-code\STEM\ecg_seizure_detection_lstm_fixed.h5"
try:
    model = load_model(model_filename)
    print("Model loaded from file.")
except (OSError, ValueError):
    model = build_lstm_model()
    print("Creating new LSTM model.")

def f1_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=BinaryFocalCrossentropy(gamma=2.0),
    metrics=['accuracy', 'Precision', 'Recall', 'AUC', f1_metric]
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=25,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr]
)

model.save(model_filename)

# --- Model Evaluation ---
test_predictions = (model.predict(test_ecgs.reshape(-1, time_steps, 1)) > 0.5).astype("int32")
cm = confusion_matrix(test_labels, test_predictions)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(test_labels, test_predictions))
