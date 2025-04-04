import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import tensorflow as tf
import webbrowser

# Load the trained Yoga Pose model
model = tf.keras.models.load_model("yoga_pose_classifier.keras")

# Define class names (Ensure they match your training dataset labels)
CLASS_NAMES = ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize Tkinter
root = tk.Tk()
root.title("Yoga Pose Analyzer")
root.geometry("800x600")

notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both')

# Home Tab
home_tab = ttk.Frame(notebook)
notebook.add(home_tab, text="Home")

label = tk.Label(home_tab)
label.pack()
feedback_label = tk.Label(home_tab, text="Press Start to begin", font=("Arial", 14), fg="blue")
feedback_label.pack()

cap = None
running = False

def predict_pose(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(resized_frame, axis=0)
    img_array = img_array.astype("float32") / 255.0
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return CLASS_NAMES[predicted_class_index]

def update_frame():
    global cap, running
    if cap is None or not running:
        return
    
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        predicted_pose = predict_pose(frame)
        feedback_label.config(text=f"Predicted Pose: {predicted_pose}")
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
    
    root.after(10, update_frame)

def start_camera():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        update_frame()

def stop_camera():
    global cap, running
    running = False
    if cap:
        cap.release()
        cap = None
    label.config(image='')
    feedback_label.config(text="Press Start to begin")

start_button = tk.Button(home_tab, text="Start", command=start_camera, bg="green", fg="white", font=("Arial", 12))
start_button.pack()
stop_button = tk.Button(home_tab, text="Stop", command=stop_camera, bg="red", fg="white", font=("Arial", 12))
stop_button.pack()

# Support Us Tab
support_tab = ttk.Frame(notebook)
notebook.add(support_tab, text="Support Us")

def open_amazon():
    webbrowser.open("https://www.amazon.com")

tk.Label(support_tab, text="Support us by shopping through this link!", font=("Arial", 14)).pack()
button = tk.Button(support_tab, text="Go to Amazon", command=open_amazon, bg="blue", fg="white", font=("Arial", 12))
button.pack()

# Lessons Tab
lessons_tab = ttk.Frame(notebook)
notebook.add(lessons_tab, text="Lessons")

lesson_label = tk.Label(lessons_tab, text="Yoga lessons will be added soon!", font=("Arial", 14))
lesson_label.pack()

root.mainloop()
