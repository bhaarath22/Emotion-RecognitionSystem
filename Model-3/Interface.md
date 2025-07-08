# Graphical User Interface For  Emotion Recognition System
### YOu can find [**Code here**](3-Interface.py).
## 1. Import Statements

```python
import random
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os
```

- **random**: Used for selecting random emotion messages
- **json**: For loading emotion messages from a JSON file
- **tkinter**: Main GUI framework
- **PIL (Pillow)**: For image processing and display
- **cv2 (OpenCV)**: For computer vision tasks (face detection, image processing)
- **numpy**: For numerical operations
- **tensorflow**: For loading and using the pre-trained emotion recognition model
- **os**: For file path operations
---
## 2. Model Loading

```python
model = None
try:
    model_path = "/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/ERSFinal/Main/ERS/EmotionRecognitionSystemFinal1.h5"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        print("Model path not provided or model not found. Running in demo mode.")
except Exception as e:
    print(f"Error loading model: {e}. Running in demo mode.")
```

- Attempts to load a pre-trained TensorFlow model from a specified path
- If the model isn't found or fails to load, the system runs in "demo mode" with placeholder predictions
- The model is expected to be a Keras model saved in .h5 format
---
## 3. Prediction Functions

### Placeholder Prediction (Demo Mode)
```python
def predict_emotion_placeholder(face):
    return "No Model Loaded", 0
```

- Returns a default message when no model is loaded
- Used as a fallback when the real model isn't available

### Main Prediction Function
```python
def predict_emotion(face):
    if model:
        face = cv2.resize(face, (256, 256))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)
        predictions = model.predict(face)
        emotion = np.argmax(predictions)
        accuracy = np.max(predictions) * 100
        return emotion, accuracy
    else:
        return predict_emotion_placeholder(face)
```

1. Checks if a model is loaded
2. Preprocesses the face image:
   - Resizes to 256x256 pixels (expected by the model)
   - Normalizes pixel values to [0,1] range
   - Adds batch dimension
3. Makes prediction using the model
4. Returns:
   - Predicted emotion (index of highest probability class)
   - Prediction confidence (as percentage)
---
## 4. GUI Setup

### Main Window Configuration
```python
root = tk.Tk()
root.title("Accuracy&loss Interface")
root.attributes('-fullscreen', True)  # Fullscreen mode

# Get screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
```

- Creates the main Tkinter window
- Sets title and fullscreen mode
- Gets screen dimensions for responsive layout

### Background Image
```python
background_img = Image.open("path/to/image.png")
background_img = background_img.resize((screen_width, screen_height), Image.LANCZOS)
background_photo = ImageTk.PhotoImage(background_img)
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
```

- Loads and resizes a background image to fill the screen
- Uses high-quality LANCZOS resampling
- Places the image as a label covering the entire window

### Camera Feed Display
```python
camera_label = tk.Label(root, bg="white")
camera_label.place(relx=0.40, rely=0.05, anchor="n", width=500, height=400)
```

- Creates a label widget to display webcam feed or uploaded images
- Positioned near the top-center of the screen
- White background when empty
---
## 5. Webcam Functions

### Start Webcam
```python
def start_webcam():
    global cap
    cap = cv2.VideoCapture(0)

    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (500, 400))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            camera_label.configure(image=frame_tk)
            camera_label.image = frame_tk
        camera_label.after(10, update_frame)

    update_frame()
```

1. Initializes video capture from default camera (index 0)
2. Defines a nested function `update_frame()` that:
   - Captures a frame from the webcam
   - Resizes and converts color space (BGR→RGB for Tkinter)
   - Converts to PhotoImage and updates the display label
   - Schedules itself to run again after 10ms (≈100 FPS)
3. Starts the update loop

### Stop Webcam
```python
def stop_webcam_and_clear_image():
    global cap
    if cap.isOpened():
        cap.release()
    camera_label.configure(image='')
    label_var.set("Emotion: ")
```

- Releases the video capture resource
- Clears the camera display label
- Resets the emotion label text
---
## 6. Image Processing Functions

### Upload Image
```python
def upload_image():
    path = filedialog.askopenfilename()
    if path:
        img = cv2.imread(path)
        img = cv2.resize(img, (256, 256))
        process_and_display_image(img)
```

- Opens a file dialog for image selection
- Reads and resizes the selected image
- Passes to processing function

### Face Detection
```python
def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return image[y:y + h, x:x + w], faces[0]
    return None, None
```

- Uses OpenCV's Haar cascade classifier for face detection
- Returns the first detected face region and its coordinates
- Returns None if no faces are detected

### Display Results
```python
def display_results(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)
    camera_label.configure(image=image_tk)
    camera_label.image = image_tk
```

- Converts image to RGB format
- Creates a Tkinter-compatible PhotoImage
- Updates the camera display label
---
## 7. Message Display System

### Message Box Setup
```python
message_box = tk.Text(root, height=10, width=55, font=("Times New Roman", 14),
                     state=tk.DISABLED, bg="white", fg="black"))
message_box.place(relx=0.85, rely=0.1, anchor="n")
message_box.tag_configure("title", font=("Times New Roman", 18, "bold"), justify='center')
```

- Creates a read-only text widget for displaying results
- Configured with specific font and colors
- Positioned at the top-right of the screen
- Sets up a "title" tag for styled text

### Emotion Messages
```python
def load_emotion_messages(json_file_path):
    with open(json_file_path, 'r') as file:
        emotion_messages = json.load(file)
    return emotion_messages

def get_emotion_message(emotion, json_file_path):
    emotion_messages = load_emotion_messages(json_file_path)
    if emotion in emotion_messages:
        return random.choice(emotion_messages[emotion])
    else:
        return "Emotion not recognized. Please try again."
```

- Loads emotion-specific messages from a JSON file
- Returns a random message for the detected emotion
- Provides a fallback message for unknown emotions

### Update Message Box
```python
def update_message_box(emotion, accuracy):
    try:
        message_box.configure(state=tk.NORMAL)
        message_box.delete(1.0, tk.END)
        message_box.insert(tk.END, "Accuracy&loss\n", "title")
        message_box.insert(tk.END, f"Accuracy: {accuracy:.2f}%\n")
        emotion_message = get_emotion_message(emotion, json_file_path)
        message_box.insert(tk.END, emotion_message + "\n")
    finally:
        message_box.configure(state=tk.DISABLED)
```

- Clears and updates the message box with:
  - Title (styled with "title" tag)
  - Prediction accuracy (formatted to 2 decimal places)
  - Emotion-specific message
- Maintains read-only state outside updates
---
## 8. Control Buttons

```python
button_webcam = tk.Button(root, text="Start Webcam", command=start_webcam, font=("Times New Roman", 20, "bold"))
button_webcam.place(x=40, y=30)

button_upload = tk.Button(root, text="Upload Image", command=upload_image, font=("Times New Roman", 20, "bold"))
button_upload.place(x=40, y=100)

button_stop = tk.Button(root, text="Stop", command=stop_webcam_and_clear_image, font=("Times New Roman", 20, "bold"))
button_stop.place(x=40, y=170)
```

- Three buttons with consistent styling:
  1. Start Webcam: Activates live camera feed
  2. Upload Image: Opens file dialog for image selection
  3. Stop: Stops webcam and clears display
---
## 9. Main Loop

```python
root.mainloop()
```

- Starts the Tkinter event loop
- Keeps the application running and responsive

## Key Features

1. **Dual Operation Modes**:
   - Full mode with loaded model for real predictions
   - Demo mode with placeholder when model is unavailable

2. **Multiple Input Sources**:
   - Live webcam feed
   - Image file upload

3. **Comprehensive Feedback**:
   - Visual display of input with face detection overlay
   - Detailed text output including:
     - Prediction accuracy
     - Emotion classification
     - Contextual messages based on detected emotion

4. **Responsive Design**:
   - Fullscreen layout adapting to different screen sizes
   - Clear visual hierarchy

5. **Error Handling**:
   - Graceful fallback for missing model
   - Try-catch blocks for critical operations

## Workflow

1. User starts webcam or uploads an image
2. System detects faces in the input
3. Detected faces are passed to the model for emotion classification
4. Results are displayed:
   - Visual feedback with bounding box
   - Textual feedback with accuracy and emotion-specific message
5. User can switch between input methods or stop the system

This implementation provides a complete, user-friendly interface for emotion recognition with robust error handling and multiple input methods.
