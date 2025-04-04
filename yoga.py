import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

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

def analyze_pose(frame, results):
    """Analyze the pose and provide basic feedback."""
    if results.pose_landmarks:
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        shoulder_distance = abs(left_shoulder.x - right_shoulder.x)
        if shoulder_distance < 0.1:
            feedback = "Keep shoulders aligned!"
        else:
            feedback = "Good posture!"
    else:
        feedback = "No pose detected."
    return feedback

def update_frame():
    """Capture frame, process it, and update Tkinter UI."""
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)  # Flip horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Convert frame to ImageTk format
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        
        # Display feedback
        feedback = analyze_pose(frame, results)
        feedback_label.config(text=feedback)
    
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
