import random
import json
import cv2
import numpy as np
import tensorflow as tf
import os

ClassNames =['anger','contempt','disgust','fear','happy','neutral','sad','surprise']

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
        emotion_index = np.argmax(predictions)
        emotion_label = ClassNames[emotion_index]  # Map index to emotion label
        accuracy = np.max(predictions) * 100
        return emotion_label, accuracy  # Return the label instead of the index
    else:
        return predict_emotion_placeholder(face)


# Function to load emotion messages from a JSON file
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


# Function to detect face
def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return image[y:y + h, x:x + w], faces[0]  # Return the face and its coordinates
    return None, None


# Function to detect face bounding box
def detect_face_box(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        return faces[0]  # Return the coordinates of the first detected face
    return (0, 0, 0, 0)  # Return a zero box if no face is detected


# Function to process image and print result
def process_image(image_path, json_file_path):
    image = cv2.imread(image_path)
    face, _ = detect_face(image)
    if face is not None:
        emotion, accuracy = predict_emotion(face)
        print(f"Emotion: {emotion}, Accuracy: {accuracy:.2f}%")
        emotion_message = get_emotion_message(emotion, json_file_path)
        print(f"Emotion Message: {emotion_message}")
    else:
        print("No face detected in the image.")


# Function to start webcam and predict emotion
def start_webcam(json_file_path):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face, _ = detect_face(frame)
        if face is not None:
            emotion, accuracy = predict_emotion(face)
            print(f"Emotion: {emotion}, Accuracy: {accuracy:.2f}%")
            emotion_message = get_emotion_message(emotion, json_file_path)
            print(f"Emotion Message: {emotion_message}")
        else:
            print("No face detected in the webcam feed.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Main function to handle either webcam or image upload
def main():
    json_file_path = '/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/ERSFinal/Main/ERS/LinesForEmotions.json'
    user_input = input("Enter 'webcam' to use the webcam or 'image' to upload an image: ").strip().lower()

    if user_input == "webcam":
        start_webcam(json_file_path)
    elif user_input == "image":
        image_path = input("Enter the path to the image: ").strip()
        if os.path.exists(image_path):
            process_image(image_path, json_file_path)
        else:
            print("Invalid image path.")
    else:
        print("Invalid input. Please enter 'webcam' or 'image'.")

if __name__ == "__main__":
    main()
