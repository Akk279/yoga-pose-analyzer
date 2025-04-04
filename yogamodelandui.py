# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# import tkinter as tk
# from PIL import Image, ImageTk

# # Load the trained yoga pose classification model
# model = tf.keras.models.load_model("yoga_pose_classifier.keras")

# # Define the class labels (Update this based on your model training)
# CLASS_NAMES = ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"]

# # Initialize MediaPipe Pose
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_drawing = mp.solutions.drawing_utils

# # Initialize Tkinter
# root = tk.Tk()
# root.title("Yoga Pose Analyzer")

# # Create a Label to display the webcam feed
# label = tk.Label(root)
# label.pack()

# # OpenCV Video Capture
# cap = cv2.VideoCapture(0)

# def preprocess_frame(frame):
#     """Preprocess frame for model input (resize and normalize)."""
#     img = cv2.resize(frame, (224, 224))  # Resize to model input size
#     img = img.astype("float32") / 255.0  # Normalize
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# def predict_pose(frame):
#     """Pass the frame to the model and return the predicted pose."""
#     processed_img = preprocess_frame(frame)
#     predictions = model.predict(processed_img)
#     predicted_class_index = np.argmax(predictions, axis=1)[0]
#     predicted_class = CLASS_NAMES[predicted_class_index]  # Get class label
#     return predicted_class

# def update_frame():
#     """Capture frame, process it, and update Tkinter UI."""
#     ret, frame = cap.read()
#     if ret:
#         frame = cv2.flip(frame, 1)  # Flip horizontally
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(rgb_frame)

#         if results.pose_landmarks:
#             mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
#         # Predict pose using the model
#         predicted_pose = predict_pose(frame)
        
#         # Convert frame to ImageTk format
#         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         imgtk = ImageTk.PhotoImage(image=img)
#         label.imgtk = imgtk
#         label.configure(image=imgtk)
        
#         # Display predicted pose
#         feedback_label.config(text=f"Predicted Pose: {predicted_pose}")
    
#     root.after(10, update_frame)  # Repeat after 10ms

# # Add Feedback Label
# feedback_label = tk.Label(root, text="Initializing...", font=("Arial", 14), fg="blue")
# feedback_label.pack()

# # Start updating frames
# update_frame()

# # Run Tkinter loop
# root.mainloop()

# # Release resources
# cap.release()
# cv2.destroyAllWindows()



import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import tensorflow as tf

# Load the trained Yoga Pose model
model = tf.keras.models.load_model("yoga_pose_classifier.keras")

# Define class names (Ensure they match your training dataset labels)
CLASS_NAMES = ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"]  # Modify as needed

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize Tkinter
root = tk.Tk()
root.title("Yoga Pose Analyzer")

# Create a Label to display the webcam feed
label = tk.Label(root)
label.pack()

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

def predict_pose(frame):
    """Preprocess frame and predict yoga pose using the trained model."""
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to model input size
    img_array = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
    img_array = img_array.astype("float32") / 255.0  # Normalize pixel values
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = CLASS_NAMES[predicted_class_index]
    
    return predicted_class

def update_frame():
    """Capture frame, process it, and update Tkinter UI."""
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)  # Flip horizontally for mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Pose
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Predict pose using the frame (instead of landmarks)
        predicted_pose = predict_pose(frame)
        feedback_label.config(text=f"Predicted Pose: {predicted_pose}")
        
        # Convert frame to ImageTk format
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
    
    root.after(10, update_frame)  # Repeat after 10ms

# Add Feedback Label
feedback_label = tk.Label(root, text="Initializing...", font=("Arial", 14), fg="blue")
feedback_label.pack()

# Start updating frames
update_frame()

# Run Tkinter loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
