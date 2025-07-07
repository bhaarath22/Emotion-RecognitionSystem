import random
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os

# Load model if available
model = None
try:
    model_path = "/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/ERSFinal/Main/ERS/EmotionRecognitionSystemFinal1.h5"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        print("Model path not provided or model not found. Running in demo mode.")
except Exception as e:
    print(f"Error loading model: {e}. Running in demo mode.")


# Placeholder prediction if no model is available
def predict_emotion_placeholder(face):
    return "No Model Loaded", 0


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


# Initialize the main Tkinter window
root = tk.Tk()
root.title("Accuracy&loss Interface")
root.attributes('-fullscreen', True)  # Set the window to fullscreen

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Load and set the background image
background_img = Image.open(
"/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/ERSFinal/Main/Data/Screenshot 2024-11-03 at 12.27.26 PM.png")
background_img = background_img.resize((screen_width, screen_height), Image.LANCZOS)
background_photo = ImageTk.PhotoImage(background_img)

# Add the background image label
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Initialize camera feed placeholder and video capture
camera_label = tk.Label(root, bg="white")
camera_label.place(relx=0.40, rely=0.05, anchor="n", width=500, height=400)  # Resized to half of original

# Define label variable for emotion output
label_var = tk.StringVar()

# Function to display the webcam feed
def start_webcam():
    global cap
    cap = cv2.VideoCapture(0)

    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (500, 400))  # Adjusted size
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            camera_label.configure(image=frame_tk)
            camera_label.image = frame_tk
        camera_label.after(10, update_frame)

    update_frame()


# Function to stop the webcam and clear the camera label
def stop_webcam_and_clear_image():
    global cap
    if cap.isOpened():
        cap.release()  # Stop the webcam
    camera_label.configure(image='')  # Clear the camera label
    label_var.set("Emotion: ")  # Reset emotion label


# Function to upload and process an image
def upload_image():
    path = filedialog.askopenfilename()
    if path:
        img = cv2.imread(path)
        img = cv2.resize(img, (256, 256))
        process_and_display_image(img)


# Process the image and display it with a bounding box and emotion label
def process_and_display_image(image):
    face = detect_face(image)
    if face is not None:
        emotion, accuracy = predict_emotion(face)
        x, y, w, h = detect_face_box(image)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{emotion} ({accuracy:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Update the message box with the detected emotion and accuracy
        update_message_box(emotion, accuracy)
    else:
        # If no face is detected, indicate it
        cv2.putText(image, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    display_results(image)


def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return image[y:y + h, x:x + w], faces[0]  # Return the face and its coordinates
    return None, None


def detect_face_box(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        return faces[0]  # Return the coordinates of the first detected face
    return (0, 0, 0, 0)  # Return a zero box if no face is detected


def display_results(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)
    camera_label.configure(image=image_tk)
    camera_label.image = image_tk


# Create a Text widget for displaying messages (read-only)
message_box = tk.Text(root, height=10, width=55, font=("Times New Roman", 14),
                       state=tk.DISABLED, bg="white", fg="black")  # Set text color to black
message_box.place(relx=0.85, rely=0.1, anchor="n")  # Position it on the top right

# Configure tags for styling
message_box.tag_configure("title", font=("Times New Roman", 18, "bold"), justify='center')

# Enable the message box to insert text
message_box.configure(state=tk.NORMAL)

# Insert initial messages
message_box.insert(tk.END, "Accuracy&loss System\n", "title")  # Title with tag
message_box.insert(tk.END, "Accuracy: 0.00%\n")  # Initial accuracy line
message_box.insert(tk.END, "Emotion will display here...\n")  # Initial emotion message

message_box.configure(state=tk.DISABLED)  # Set back to read-only


## Function to load emotion messages from a JSON file
def load_emotion_messages(json_file_path):
    with open(json_file_path, 'r') as file:
        emotion_messages = json.load(file)
    return emotion_messages

# Function to get the emotion message based on the predicted emotion
def get_emotion_message(emotion, json_file_path):
    # Load emotion messages from JSON file
    emotion_messages = load_emotion_messages(json_file_path)

    # If the emotion exists in the dictionary, return a random message from the list
    if emotion in emotion_messages:
        return random.choice(emotion_messages[emotion])
    else:
        return "Emotion not recognized. Please try again."

# Example usage
json_file_path = '/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/ERSFinal/Main/ERS/LinesForEmotions.json'
# Function to update the message box with the detected emotion and accuracy
def update_message_box(emotion, accuracy):
    try:
        message_box.configure(state=tk.NORMAL)  # Enable the message box to update
        message_box.delete(1.0, tk.END)  # Clear the message box
        message_box.insert(tk.END, "Accuracy&loss\n", "title")  # Title with tag
        message_box.insert(tk.END, f"Accuracy: {accuracy:.2f}%\n")  # Insert accuracy

        # Get the message based on the detected emotion
        emotion_message = get_emotion_message(emotion, json_file_path)
        message_box.insert(tk.END, emotion_message + "\n")  # Insert emotion message

    except Exception as e:
        message_box.insert(tk.END, f"Error updating message: {str(e)}\n")  # Display error
    finally:
        message_box.configure(state=tk.DISABLED)  # Set back to read-only

button_webcam = tk.Button(
    root,
    text="Start Webcam",
    command=start_webcam,
    font=("Times New Roman", 20, "bold")
)
button_webcam.place(x=40, y=30)

button_upload = tk.Button(
    root,
    text="Upload Image",
    command=upload_image,
    font=("Times New Roman", 20, "bold")
)
button_upload.place(x=40, y=100)

button_stop = tk.Button(
    root,
    text="Stop",
    command=stop_webcam_and_clear_image,
    font=("Times New Roman", 20, "bold")
)
button_stop.place(x=40, y=170)

root.mainloop()
