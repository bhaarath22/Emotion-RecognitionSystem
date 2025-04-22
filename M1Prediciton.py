import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# Loading the pre-trained emotion recognition model
model = load_model('/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/ERNDA/EMR_NDA.h5')
#model = load_model('/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/ERNDA3Emotions/EmotionRecognitionSystem.h5')

# Loading Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def EmotionPrediction(image, model):
    # Converting color image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detecting faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)

    if len(faces) == 0:
        print("No face detected.")
        return image  # Returning the original image if no face is detected

    for (x, y, w, h) in faces:
        # Drawing bounding box around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Extracting face region
        face = image[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (256, 256))
        face_resized = np.expand_dims(face_resized, axis=0)
        face_resized = face_resized / 255.0

        # prediction
        prediction = model.predict(face_resized)
        predicted_label = "Happy" if np.argmax(prediction) == 1 else "Not Happy"

        # Writing predicted label above the bounding box
        cv2.putText(image, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    return image


# Webcam function with bounding box and label on top of the face
def Webcam(model):
    cap = cv2.VideoCapture(0)  # Starting webcam capture from the device's camera
    if not cap.isOpened():
        print("Error: Webcam not found!")
        return

    start_time = time.time()  # Start time
    last_pred_time = start_time  # Initialization of the last prediction time
    prediction_interval = 30  # Interval in seconds for prediction

    while True:
        ret, frame = cap.read()  # Reading a frame from the webcam
        if not ret:
            print("Failed to grab frame")
            break
        frame_with_prediction = EmotionPrediction(frame, model)
        # Showing the image with bounding box and label
        cv2.imshow("Webcam Feed", frame_with_prediction)
        # Checking if 30 seconds have passed since the last prediction
        current_time = time.time()
        if current_time - last_pred_time >= prediction_interval:
            print("Prediction Updated")
            last_pred_time = current_time  # Update the last prediction time
        # To exit from the webcam feed, press the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# Image path case
def ImagePrediction(image_path, model):
    image = cv2.imread(image_path)
    # Emotion prediction and draw bounding boxes and labels
    image_with_prediction = EmotionPrediction(image, model)

    # To Show the image with bounding box and label
    cv2.imshow("Image with Emotion", image_with_prediction)
    # To exit the window, press any key
    cv2.waitKey(0)
    cv2.destroyAllWindows()

use_webcam = True  # False for image path
if use_webcam:
    Webcam(model)
else:
    image_path = ''  # path
    ImagePrediction(image_path, model)
