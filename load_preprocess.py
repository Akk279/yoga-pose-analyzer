import kagglehub
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Download latest version of dataset
path = kagglehub.dataset_download("niharika41298/yoga-poses-dataset")
print("Path to dataset files:", path)

# Define dataset directories
dataset_dir = "F:/vs studio code/yoga/DATASET"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")



import os
from PIL import Image, ImageFile

# Allow loading of truncated images to prevent crashes
ImageFile.LOAD_TRUNCATED_IMAGES = True

dataset_path = "F:/vs studio code/yoga/DATASET"

def check_and_remove_corrupted_images(directory):
    """Check for corrupted images and remove them"""
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify if the image is valid
            except (IOError, SyntaxError):
                print(f"Corrupted image found and removed: {file_path}")
                os.remove(file_path)  # Delete corrupted file

# Run the function to clean dataset
check_and_remove_corrupted_images(dataset_path)

print("Dataset cleaned. Now you can safely train your model.")

# Now you can run your training script without issues
# os.system("python -u 'f:/vs studio code/yoga/load_preprocess.py'")



# Data Preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training')
val_generator = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation')

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_generator, validation_data=val_generator, epochs=10)


import matplotlib.pyplot as plt

history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# Plot Accuracy & Loss
plt.figure(figsize=(10,5))

# Accuracy Plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.show()




# Save Model
model.save("yoga_pose_classifier.keras")
print("Model training complete and saved!")

