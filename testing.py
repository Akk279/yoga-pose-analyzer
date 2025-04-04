import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

test_folder = "F:/vs studio code/yoga/DATASET/TEST"

# Recursively get image paths from all subfolders
image_files = []
for root, _, files in os.walk(test_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(root, file))

print("Found images:", image_files)  # Debugging step

if not image_files:
    print("No valid images found in the TEST folder.")
else:
    # Load the trained model
    model = tf.keras.models.load_model("yoga_pose_classifier.keras")  # Ensure you saved the model as .keras

    # Select the first image for testing
    img_path = image_files[0]
    print(f"Testing image: {img_path}")

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    print(f"Predicted class: {predicted_class}")  # Map this to actual class labels if needed
