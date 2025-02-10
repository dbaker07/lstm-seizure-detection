import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

# Enable eager execution for better error tracking
tf.config.run_functions_eagerly(True)

# Function to load and preprocess spectrogram images from a folder
def load_and_preprocess_spectrograms_from_folder(folder_path, img_size=(128, 128)):
    images = []
    labels = []
    
    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            # Construct the full path of the image
            img_path = os.path.join(folder_path, filename)
            
            # Load the image with the target size
            img = image.load_img(img_path, target_size=img_size, color_mode='grayscale')
            img_array = image.img_to_array(img)
            images.append(img_array)
            
            # Assign label based on filename (seizure vs. no seizure)
            if '- no seizure' in filename.lower():
                labels.append(0)  # No Seizure
            else:
                labels.append(1)  # Seizure
    
    images = np.array(images)
    labels = np.array(labels)
    images = images / 255.0  # Normalize images
    return images, labels

# Example data (replace with the actual folder path)
folder_path = r"C:\Users\david\STEM Project\Graph JPGs"  # Replace with your images folder path

# Load and preprocess images from the folder
X, y = load_and_preprocess_spectrograms_from_folder(folder_path)

# Check the shapes of the images and labels
print(f"x shape: {X.shape}")
print(f"y shape: {y.shape}")

# Visualize a few images to check correctness
for i in range(3):
    plt.imshow(X[i].squeeze(), cmap='gray')
    plt.show()

# Split data into training and testing sets
tf.random.set_seed(42)
indices = tf.random.shuffle(tf.range(tf.shape(X)[0]))
test_size = int(0.2 * len(X))
train_size = len(X) - test_size
train_indices = indices[:train_size]
test_indices = indices[train_size:]
x_train, y_train = tf.gather(X, train_indices), tf.gather(y, train_indices)
x_test, y_test = tf.gather(X, test_indices), tf.gather(y, test_indices)

# Check the shapes of train/test sets
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Compute class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(class_weights))

# Build the CNN model
def build_seizure_prediction_model(input_shape=(128, 128, 1)):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout to prevent overfitting
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary output
    return model

# Define the model
model = build_seizure_prediction_model()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test), 
                    class_weight=class_weights, callbacks=[early_stopping])

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot the training and validation loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Confusion matrix and classification report
y_pred = (model.predict(x_test) > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Making predictions
new_spectrogram = np.expand_dims(x_test[0], axis=0)  # Add batch dimension
prediction = model.predict(new_spectrogram)
print("Prediction (0: no seizure, 1: seizure):", prediction[0][0])
