# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import tensorflow as tf
# # import tkinter as tk
# # from PIL import Image, ImageTk

# # # Load the trained yoga pose classification model
# # model = tf.keras.models.load_model("yoga_pose_classifier.keras")

# # # Define the class labels (Update this based on your model training)
# # CLASS_NAMES = ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"]

# # # Initialize MediaPipe Pose
# # mp_pose = mp.solutions.pose
# # pose = mp_pose.Pose()
# # mp_drawing = mp.solutions.drawing_utils

# # # Initialize Tkinter
# # root = tk.Tk()
# # root.title("Yoga Pose Analyzer")

# # # Create a Label to display the webcam feed
# # label = tk.Label(root)
# # label.pack()

# # # OpenCV Video Capture
# # cap = cv2.VideoCapture(0)

# # def preprocess_frame(frame):
# #     """Preprocess frame for model input (resize and normalize)."""
# #     img = cv2.resize(frame, (224, 224))  # Resize to model input size
# #     img = img.astype("float32") / 255.0  # Normalize
# #     img = np.expand_dims(img, axis=0)  # Add batch dimension
# #     return img

# # def predict_pose(frame):
# #     """Pass the frame to the model and return the predicted pose."""
# #     processed_img = preprocess_frame(frame)
# #     predictions = model.predict(processed_img)
# #     predicted_class_index = np.argmax(predictions, axis=1)[0]
# #     predicted_class = CLASS_NAMES[predicted_class_index]  # Get class label
# #     return predicted_class

# # def update_frame():
# #     """Capture frame, process it, and update Tkinter UI."""
# #     ret, frame = cap.read()
# #     if ret:
# #         frame = cv2.flip(frame, 1)  # Flip horizontally
# #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         results = pose.process(rgb_frame)

# #         if results.pose_landmarks:
# #             mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
# #         # Predict pose using the model
# #         predicted_pose = predict_pose(frame)
        
# #         # Convert frame to ImageTk format
# #         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# #         imgtk = ImageTk.PhotoImage(image=img)
# #         label.imgtk = imgtk
# #         label.configure(image=imgtk)
        
# #         # Display predicted pose
# #         feedback_label.config(text=f"Predicted Pose: {predicted_pose}")
    
# #     root.after(10, update_frame)  # Repeat after 10ms

# # # Add Feedback Label
# # feedback_label = tk.Label(root, text="Initializing...", font=("Arial", 14), fg="blue")
# # feedback_label.pack()

# # # Start updating frames
# # update_frame()

# # # Run Tkinter loop
# # root.mainloop()

# # # Release resources
# # cap.release()
# # cv2.destroyAllWindows()



# import cv2
# import mediapipe as mp
# import numpy as np
# import tkinter as tk
# from PIL import Image, ImageTk
# import tensorflow as tf

# # Load the trained Yoga Pose model
# model = tf.keras.models.load_model("yoga_pose_classifier.keras")

# # Define class names (Ensure they match your training dataset labels)
# CLASS_NAMES = ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"]  # Modify as needed

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

# def predict_pose(frame):
#     """Preprocess frame and predict yoga pose using the trained model."""
#     resized_frame = cv2.resize(frame, (224, 224))  # Resize to model input size
#     img_array = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
#     img_array = img_array.astype("float32") / 255.0  # Normalize pixel values
    
#     predictions = model.predict(img_array)
#     predicted_class_index = np.argmax(predictions, axis=1)[0]
#     predicted_class = CLASS_NAMES[predicted_class_index]
    
#     return predicted_class

# def update_frame():
#     """Capture frame, process it, and update Tkinter UI."""
#     ret, frame = cap.read()
#     if ret:
#         frame = cv2.flip(frame, 1)  # Flip horizontally for mirror effect
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Process with MediaPipe Pose
#         results = pose.process(rgb_frame)
#         if results.pose_landmarks:
#             mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
#         # Predict pose using the frame (instead of landmarks)
#         predicted_pose = predict_pose(frame)
#         feedback_label.config(text=f"Predicted Pose: {predicted_pose}")
        
#         # Convert frame to ImageTk format
#         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         imgtk = ImageTk.PhotoImage(image=img)
#         label.imgtk = imgtk
#         label.configure(image=imgtk)
    
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
from tkinter import ttk
from PIL import Image, ImageTk
import tensorflow as tf
import webbrowser

# Load the trained Yoga Pose model
try:
    model = tf.keras.models.load_model("yoga_pose_classifier.keras")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define class names and colors
CLASS_NAMES = ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"]
CLASS_COLORS = {
    "Downdog": "#FF7F50",
    "Goddess": "#9370DB",
    "Plank": "#20B2AA",
    "Tree": "#32CD32",
    "Warrior2": "#4169E1"
}

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Main application class
class YogaPoseAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("ANT Yoga - Pose Analyzer")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f8f9fa")
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure("TNotebook", background="#f8f9fa", borderwidth=0)
        self.style.configure("TNotebook.Tab", 
                            font=("Arial", 12, "bold"), 
                            padding=[20, 10],
                            background="#e9ecef",
                            foreground="#495057")
        self.style.map("TNotebook.Tab",
                      background=[("selected", "#ffffff")],
                      foreground=[("selected", "#212529")])
        
        # Create main container
        self.main_frame = tk.Frame(root, bg="#f8f9fa")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_home_tab()
        self.create_analyzer_tab()
        self.create_lessons_tab()
        self.create_support_tab()
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.is_camera_active = False
        
    def create_home_tab(self):
        """Create the home/welcome tab"""
        home_tab = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(home_tab, text=" Home ")
        
        # Header
        header_frame = tk.Frame(home_tab, bg="#6c757d", height=200)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(header_frame, 
                             text="ANTYoga Pose Analyzer", 
                             font=("Arial", 28, "bold"), 
                             fg="white", 
                             bg="#6c757d")
        title_label.pack(pady=40)
        
        # Content
        content_frame = tk.Frame(home_tab, bg="#ffffff")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=50, pady=30)
        
        # Welcome message
        welcome_text = """Welcome to ANTYoga - your personal yoga assistant!

This application helps you perfect your yoga poses with real-time feedback. 

Features:
- Real-time pose detection and analysis
- Confidence scoring for your poses
- Instructional videos for each pose
- Personalized feedback and tips

Get started by clicking on the 'Analyzer' tab to begin your yoga session!"""
        
        welcome_label = tk.Label(content_frame, 
                               text=welcome_text,
                               font=("Arial", 14),
                               bg="#ffffff",
                               justify=tk.LEFT)
        welcome_label.pack(pady=20, anchor=tk.W)
        
        # Get started button
        start_button = tk.Button(content_frame,
                               text="Start Analyzing →",
                               font=("Arial", 14, "bold"),
                               bg="#28a745",
                               fg="white",
                               command=lambda: self.notebook.select(1),
                               padx=20,
                               pady=10)
        start_button.pack(pady=30)
        
    def create_analyzer_tab(self):
        """Create the main analyzer tab with camera feed"""
        analyzer_tab = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(analyzer_tab, text=" Analyzer ")
        
        # Create two-column layout
        left_frame = tk.Frame(analyzer_tab, bg="#ffffff")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        right_frame = tk.Frame(analyzer_tab, bg="#ffffff")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
        
        # Camera feed
        cam_frame = tk.LabelFrame(left_frame, 
                                text="Camera Feed", 
                                font=("Arial", 12, "bold"),
                                bg="#ffffff",
                                padx=10,
                                pady=10)
        cam_frame.pack(fill=tk.BOTH, expand=True)
        
        self.cam_label = tk.Label(cam_frame, bg="black")
        self.cam_label.pack(padx=10, pady=10)
        
        # Control buttons
        control_frame = tk.Frame(left_frame, bg="#ffffff")
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = tk.Button(control_frame,
                                    text="Start Camera",
                                    command=self.toggle_camera,
                                    bg="#007bff",
                                    fg="white",
                                    font=("Arial", 12))
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.capture_button = tk.Button(control_frame,
                                     text="Capture Pose",
                                     command=self.take_screenshot,
                                     state=tk.DISABLED,
                                     bg="#17a2b8",
                                     fg="white",
                                     font=("Arial", 12))
        self.capture_button.pack(side=tk.LEFT, padx=5)
        
        # Pose information
        info_frame = tk.LabelFrame(right_frame,
                                 text="Pose Information",
                                 font=("Arial", 12, "bold"),
                                 bg="#ffffff",
                                 padx=10,
                                 pady=10)
        info_frame.pack(fill=tk.BOTH)
        
        # Current pose display
        pose_frame = tk.Frame(info_frame, bg="#ffffff")
        pose_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(pose_frame, 
               text="Current Pose:", 
               font=("Arial", 14), 
               bg="#ffffff").pack(side=tk.LEFT)
        
        self.pose_value_label = tk.Label(pose_frame, 
                                      text="None", 
                                      font=("Arial", 24, "bold"), 
                                      bg="#ffffff")
        self.pose_value_label.pack(side=tk.LEFT, padx=10)
        
        # Confidence meter
        confidence_frame = tk.Frame(info_frame, bg="#ffffff")
        confidence_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(confidence_frame, 
               text="Confidence:", 
               font=("Arial", 14), 
               bg="#ffffff").pack(side=tk.LEFT)
        
        self.confidence_meter = ttk.Progressbar(confidence_frame, 
                                              orient="horizontal", 
                                              length=200, 
                                              mode="determinate")
        self.confidence_meter.pack(side=tk.LEFT, padx=10)
        
        self.confidence_label = tk.Label(confidence_frame, 
                                      text="0%", 
                                      font=("Arial", 14), 
                                      bg="#ffffff")
        self.confidence_label.pack(side=tk.LEFT)
        
        # Pose tips
        tips_frame = tk.LabelFrame(info_frame,
                                 text="Pose Tips",
                                 font=("Arial", 12, "bold"),
                                 bg="#ffffff",
                                 padx=10,
                                 pady=10)
        tips_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tips_text = tk.Text(tips_frame, 
                               height=10, 
                               width=40, 
                               font=("Arial", 12), 
                               wrap=tk.WORD, 
                               bg="#f8f9fa",
                               padx=10,
                               pady=10)
        self.tips_text.pack(fill=tk.BOTH, expand=True)
        self.tips_text.insert(tk.END, "Tips will appear here when a pose is detected.")
        self.tips_text.config(state=tk.DISABLED)
        
    def create_lessons_tab(self):
        """Create the yoga lessons tab"""
        lessons_tab = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(lessons_tab, text=" Lessons ")
        
        # Header
        header_frame = tk.Frame(lessons_tab, bg="#6c757d", height=100)
        header_frame.pack(fill=tk.X)
        
        tk.Label(header_frame, 
               text="Yoga Lessons", 
               font=("Arial", 24, "bold"), 
               fg="white", 
               bg="#6c757d").pack(pady=20)
        
        # Content
        content_frame = tk.Frame(lessons_tab, bg="#ffffff")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=50, pady=30)
        
        # Lesson cards
        lessons = [
            {"name": "Downward Dog", "desc": "Strengthens arms and legs while stretching shoulders and hamstrings"},
            {"name": "Goddess Pose", "desc": "Opens the hips and chest while strengthening the lower body"},
            {"name": "Plank Pose", "desc": "Builds core strength and stability"},
            {"name": "Tree Pose", "desc": "Improves balance and focus while strengthening legs"},
            {"name": "Warrior II", "desc": "Strengthens legs and opens hips and chest"}
        ]
        
        for i, lesson in enumerate(lessons):
            card_frame = tk.Frame(content_frame, 
                                 bg="#e9ecef", 
                                 padx=20, 
                                 pady=15,
                                 relief=tk.RAISED,
                                 borderwidth=1)
            card_frame.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="nsew")
            
            tk.Label(card_frame, 
                   text=lesson["name"], 
                   font=("Arial", 16, "bold"), 
                   bg="#e9ecef").pack(anchor=tk.W)
            
            tk.Label(card_frame, 
                   text=lesson["desc"], 
                   font=("Arial", 12), 
                   bg="#e9ecef",
                   wraplength=300,
                   justify=tk.LEFT).pack(anchor=tk.W, pady=5)
            
            # Demo button (placeholder)
            tk.Button(card_frame,
                    text="View Demo",
                    bg="#6c757d",
                    fg="white",
                    font=("Arial", 10)).pack(anchor=tk.E)
        
        # Configure grid
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        
    def create_support_tab(self):
        """Create the support/donate tab"""
        support_tab = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(support_tab, text=" Support Us ")
        
        # Header
        header_frame = tk.Frame(support_tab, bg="#6c757d", height=100)
        header_frame.pack(fill=tk.X)
        
        tk.Label(header_frame, 
               text="Support Our Work", 
               font=("Arial", 24, "bold"), 
               fg="white", 
               bg="#6c757d").pack(pady=20)
        
        # Content
        content_frame = tk.Frame(support_tab, bg="#ffffff")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=50, pady=30)
        
        # Support message
        support_text = """We're dedicated to providing free yoga resources to everyone!

If you find this application helpful, please consider supporting us to help cover development costs and allow us to create more features.

Your support helps us:
- Improve the pose detection algorithms
- Add more yoga poses and routines
- Create detailed instructional content
- Maintain and update the application

Thank you for being part of our yoga community!"""
        
        tk.Label(content_frame, 
               text=support_text,
               font=("Arial", 14),
               bg="#ffffff",
               justify=tk.LEFT,
               wraplength=600).pack(pady=20)
        
        # Support options
        options_frame = tk.Frame(content_frame, bg="#ffffff")
        options_frame.pack(pady=30)
        
        # Amazon affiliate link button
        amazon_button = tk.Button(options_frame,
                                text="Shop Yoga Gear on Amazon",
                                command=lambda: webbrowser.open("https://www.amazon.com/yoga-gear"),
                                bg="#FF9900",
                                fg="black",
                                font=("Arial", 14, "bold"),
                                padx=20,
                                pady=10)
        amazon_button.pack(pady=10, fill=tk.X)
        
        # Donate button
        donate_button = tk.Button(options_frame,
                                text="Make a Donation",
                                bg="#28a745",
                                fg="white",
                                font=("Arial", 14, "bold"),
                                padx=20,
                                pady=10)
        donate_button.pack(pady=10, fill=tk.X)
        
    def toggle_camera(self):
        """Start or stop the camera feed"""
        if not self.is_camera_active:
            self.is_camera_active = True
            self.start_button.config(text="Stop Camera", bg="#dc3545")
            self.capture_button.config(state=tk.NORMAL)
            self.update_frame()
        else:
            self.is_camera_active = False
            self.start_button.config(text="Start Camera", bg="#007bff")
            self.capture_button.config(state=tk.DISABLED)
    
    def update_frame(self):
        """Update the camera feed frame"""
        if self.is_camera_active:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe Pose
                results = pose.process(rgb_frame)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Predict pose
                predicted_pose, confidence = self.predict_pose(frame)
                
                # Update UI elements
                self.pose_value_label.config(text=predicted_pose, 
                                           fg=CLASS_COLORS.get(predicted_pose, "black"))
                self.confidence_meter["value"] = confidence
                self.confidence_label.config(text=f"{confidence:.0f}%")
                
                # Update tips
                self.update_tips(predicted_pose, confidence)
                
                # Convert frame to ImageTk format
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.thumbnail((640, 480))
                imgtk = ImageTk.PhotoImage(image=img)
                self.cam_label.imgtk = imgtk
                self.cam_label.configure(image=imgtk)
            
            self.root.after(30, self.update_frame)
    
    def predict_pose(self, frame):
        """Predict the yoga pose from frame"""
        resized_frame = cv2.resize(frame, (224, 224))
        img_array = np.expand_dims(resized_frame, axis=0)
        img_array = img_array.astype("float32") / 255.0
        
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100
        predicted_class = CLASS_NAMES[predicted_class_index]
        
        return predicted_class, confidence
    
    def update_tips(self, pose, confidence):
        """Update the tips text based on current pose"""
        tips = {
            "Downdog": "Tips for Downward Dog:\n- Spread your fingers wide\n- Press your heels down\n- Engage your core\n- Lengthen your spine",
            "Goddess": "Tips for Goddess Pose:\n- Keep knees over ankles\n- Engage your core\n- Relax your shoulders\n- Breathe deeply",
            "Plank": "Tips for Plank Pose:\n- Keep body in straight line\n- Engage your core\n- Don't let hips sag\n- Breathe steadily",
            "Tree": "Tips for Tree Pose:\n- Focus on a fixed point\n- Bring foot to inner thigh or calf (not knee)\n- Engage your core\n- Relax your shoulders",
            "Warrior2": "Tips for Warrior II:\n- Front knee at 90 degrees\n- Back foot at 45 degrees\n- Arms parallel to floor\n- Gaze over front hand"
        }
        
        self.tips_text.config(state=tk.NORMAL)
        self.tips_text.delete(1.0, tk.END)
        
        if pose in tips:
            self.tips_text.insert(tk.END, tips[pose])
            if confidence < 70:
                self.tips_text.insert(tk.END, f"\n\nYour pose needs work. Try to improve alignment.")
            elif confidence < 85:
                self.tips_text.insert(tk.END, f"\n\nGood form! Keep refining your pose.")
            else:
                self.tips_text.insert(tk.END, f"\n\nExcellent form! Maintain this alignment.")
        else:
            self.tips_text.insert(tk.END, "Assume a yoga pose to get feedback.")
        
        self.tips_text.config(state=tk.DISABLED)
    
    def take_screenshot(self):
        """Capture the current frame"""
        if self.is_camera_active:
            ret, frame = self.cap.read()
            if ret:
                filename = f"yoga_pose_{CLASS_NAMES[0]}.png"  # In a real app, use current pose and timestamp
                cv2.imwrite(filename, cv2.flip(frame, 1))
                print(f"Screenshot saved as {filename}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = YogaPoseAnalyzer(root)
    root.mainloop()
    
    # Release resources
    if app.cap.isOpened():
        app.cap.release()
    cv2.destroyAllWindows()