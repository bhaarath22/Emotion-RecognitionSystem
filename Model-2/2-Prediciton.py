import cv2
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'EmotionRecognitionSystem.h5'
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# Preprocess image
def preprocess_image(frame):
    frame_resized = cv2.resize(frame, (200, 200))
    img_array = tf.keras.preprocessing.image.img_to_array(frame_resized)
    img_array /= 255.0  # Normalizing the image
    img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension
    return img_array


# Display emotion label and message on the webcam feed
def display_emotion_info(frame, emotion, x, y):
    # Emotion message based on detected emotion
    if emotion == "happy":
        message = "It's great to see you smiling! Keep spreading those positive vibes."
    elif emotion == "sad":
        message = "I see that you’re feeling a bit down. It’s okay to feel sad sometimes. Remember, it's okay to take a break and reach out if you need support."
    elif emotion == "angry":
        message = "I notice you might be feeling angry. Take a deep breath—it’s okay to feel this way. Maybe try some relaxation exercises to calm down."
    else:
        message = "Emotion not recognized, but I hope you're feeling okay!"

    # Display the emotion label above the face
    cv2.putText(frame, f"Emotion: {emotion}", (x, y-40 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the emotion message below the face
    cv2.putText(frame, message, (40, 50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


# Prediction function to handle both webcam and image input
def predict_emotion(model, use_webcam=True, image_path=None):
    if use_webcam:
        # Open webcam feed
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        last_emotion = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            # Preprocess the image for emotion prediction
            img_array = preprocess_image(frame)
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction)
            class_labels = ["angry", "happy", "sad"]  
            predicted_class_label = class_labels[predicted_class_index]

            # If the emotion has changed, display the new message
            if predicted_class_label != last_emotion:
                last_emotion = predicted_class_label  # Update last emotion
                print(f"Emotion changed: {predicted_class_label}")  

            # Display bounding box and emotion label
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Display the emotion label and message on the webcam feed
                display_emotion_info(frame, last_emotion, x, y)

            # Show webcam feed with emotion label and message
            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        if image_path is None:
            print("Error: No image path provided")
            return

        # Load and preprocess the input image
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Could not load image")
            return

        img_array = preprocess_image(img)
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        class_labels = ["happy", "sad", "angry"] 
        predicted_class_label = class_labels[predicted_class_index]

        # Display the prediction
        print(f"Predicted Emotion: {predicted_class_label}")
        display_emotion_info(img, predicted_class_label, 1, 1)  # Display the message on the image

        # Display emotion on the image
        cv2.putText(img, f"Emotion: {predicted_class_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Predicted Emotion', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

predict_emotion(model, use_webcam=True)
# predict_emotion(model, use_webcam=False, image_path="path/to/your/image.jpg")
