'''

███████ ███    ██ ███████ ███████     ██████  ██████  ██████  ███████ ███████ 
██      ████   ██ ██      ██         ██      ██    ██ ██   ██ ██      ██      
█████   ██ ██  ██ █████   ███████    ██      ██    ██ ██   ██ █████   ███████ 
██      ██  ██ ██ ██           ██    ██      ██    ██ ██   ██ ██           ██ 
███████ ██   ████ ███████ ███████ ██  ██████  ██████  ██████  ███████ ███████ 

'''                                                                            

import os
import pydicom
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Your dataset path (The dataset format is described in the repository)
dataset_path = "C:/Users/talha/Desktop/DS1"
metadata_path = os.path.join(dataset_path, "metadata.xlsx")

# Read Excel File
data_df = pd.read_excel(metadata_path)

# Read Category Labels
def encode_labels(label):
    mapping = {1: 0, 2: 1, 4: 2, 5: 3}  # BIRADS1 -> 0, BIRADS2 -> 1, BIRADS4 -> 2, BIRADS5 -> 3
    return mapping[label]

# Preprocessing Function
def preprocess_image(dicom_path, view):
    # Read DICOM File
    dicom = pydicom.dcmread(dicom_path)
    image = dicom.pixel_array

    # Crop the Right Breast Images
    if view in ['RCC.dcm', 'RMLO.dcm']:
        image = image[:, int(image.shape[1] * 0.6):]
    # Crop the Left Breast Images
    elif view in ['LCC.dcm', 'LMLO.dcm']:
        image = image[:, :int(image.shape[1] * 0.4)]

    # Crop the Top and Bottom
    image = image[int(image.shape[0] * 0.2):int(image.shape[0] * 0.8), :]

    # Normalization
    image = cv2.resize(image, (224, 224))  
    image = image / np.max(image)  # Normalization
    return image

# Show 5 Preprocessed Images
preprocessed_images = []
for index, row in data_df.iterrows():
    if index == 5:  # Just 5 Images
        break
    category = row['Category']
    patient_id = row['Patient_id']
    file_name = row['File_name']
    view = row['File_name']
    
    # Create File Path
    image_path = os.path.join(dataset_path, f"Category{category[-1]}", str(patient_id), file_name)
    
    # Preprocessing
    processed_image = preprocess_image(image_path, view)
    preprocessed_images.append(processed_image)

# Show Images
for i, img in enumerate(preprocessed_images):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(f"Preprocessed Image {i+1}")
    plt.show()

# Split Data
X = []  # Images
y = []  # Labels (Category)

# Fill the Dataset
for index, row in data_df.iterrows():
    category = row['Category']
    patient_id = row['Patient_id']
    file_name = row['File_name']
    view = row['File_name']

    # Create the File Path
    image_path = os.path.join(dataset_path, f"Category{category[-1]}", str(patient_id), file_name)
    
    # Preprocessing
    processed_image = preprocess_image(image_path, view)
    X.append(processed_image)
    y.append(encode_labels(int(category[-1])))

# Numpy array
X = np.array(X)
y = np.array(y)

# Splitting the Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # 4 classes (0-3 range)
])

# Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

patient_predictions = {}
actual_labels = {}

for index, row in data_df.iterrows():
    patient_id = row['Patient_id']
    category = row['Category']
    file_name = row['File_name']
    view = row['File_name']

    # Create File Path
    image_path = os.path.join(dataset_path, f"Category{category[-1]}", str(patient_id), file_name)
    
    # Preprocessing
    processed_image = preprocess_image(image_path, view)
    
    # Prediction
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    predicted_category = np.argmax(prediction, axis=1)[0]
    
    if patient_id not in patient_predictions:
        patient_predictions[patient_id] = []
    patient_predictions[patient_id].append(predicted_category)
    
    # Real Labels
    if patient_id not in actual_labels:
        actual_labels[patient_id] = encode_labels(int(category[-1]))

# Predict the best Class for a Image
final_predictions = {}
for patient_id, predictions in patient_predictions.items():
    most_common_prediction = Counter(predictions).most_common(1)[0][0]
    final_predictions[patient_id] = most_common_prediction

# List of Predict and True Labels
y_true = list(actual_labels.values())
y_pred = list(final_predictions.values())

# Classification Report
print(classification_report(y_true, y_pred, target_names=["BIRADS1", "BIRADS2", "BIRADS4", "BIRADS5"]))

# Confusion Matrix (Approx: 90% Accuracy)
cm = confusion_matrix(y_true, y_pred)

# Draw Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["BIRADS1", "BIRADS2", "BIRADS4", "BIRADS5"], yticklabels=["BIRADS1", "BIRADS2", "BIRADS4", "BIRADS5"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Train and Validation Accuracy Graph
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Train and Validation Loss Graph
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()

'''

 ██████   ██████   ██████  ██████      ██      ██    ██  ██████ ██   ██ 
██       ██    ██ ██    ██ ██   ██     ██      ██    ██ ██      ██  ██  
██   ███ ██    ██ ██    ██ ██   ██     ██      ██    ██ ██      █████   
██    ██ ██    ██ ██    ██ ██   ██     ██      ██    ██ ██      ██  ██  
 ██████   ██████   ██████  ██████      ███████  ██████   ██████ ██   ██ 
                                                                        
                                                                        

'''