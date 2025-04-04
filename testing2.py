# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os

# # Define test folder path
# test_folder = "F:/vs studio code/yoga/DATASET/TEST"

# # Get image files from subfolders
# image_files = []
# for root, dirs, files in os.walk(test_folder):
#     for file in files:
#         if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#             image_files.append(os.path.join(root, file))  # Store full paths

# # Debugging: Print found images
# print("Filtered image files:", image_files)

# # Check if images exist
# if not image_files:
#     print("❌ No valid images found in the TEST folder.")
#     exit()

# # Load the trained model
# model = tf.keras.models.load_model("yoga_pose_classifier.keras")

# # Define yoga pose labels (Modify as per your dataset)
# pose_labels = ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"]  # Adjust if needed

# # Loop through found images for testing
# for img_path in image_files:
#     print(f"\n🔍 Testing image: {img_path}")

#     try:
#         # Load and preprocess the image
#         img = image.load_img(img_path, target_size=(224, 224))
#         img_array = image.img_to_array(img) / 255.0  # Normalize
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#         # Predict
#         prediction = model.predict(img_array)
#         predicted_class = np.argmax(prediction, axis=1)[0]

#         # Map prediction to yoga pose
#         yoga_pose = pose_labels[predicted_class] if predicted_class < len(pose_labels) else "Unknown Pose"

#         print(f"✅ Predicted Yoga Pose: {yoga_pose}")

#     except Exception as e:
#         print(f"❌ Error processing {img_path}: {e}")

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

# Define class labels (ensure these match your model's training classes)
class_labels = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']

# Load the trained model
model = tf.keras.models.load_model("yoga_pose_classifier.keras")

# Define the test folder
test_folder = "F:/vs studio code/yoga/DATASET/TEST/goddess"

# Search for images inside all subfolders
image_files = []
for root, dirs, files in os.walk(test_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(root, file))

if not image_files:
    print("No valid images found in the TEST folder.")
else:
    plt.figure(figsize=(10, 5))  # Adjust figure size

    for i, img_path in enumerate(image_files[:5]):  # Display first 5 images
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        # Display the image with predicted label
        plt.subplot(2, 3, i + 1)  # Create a grid of images
        plt.imshow(img)
        plt.axis('off')
        plt.title(predicted_label)

        print(f"Image: {os.path.basename(img_path)} → Predicted Pose: {predicted_label}")

    plt.show()  # Show all images with predictions
