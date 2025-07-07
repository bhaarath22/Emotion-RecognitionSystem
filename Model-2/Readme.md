# 🎭 Emotion Recognition System Using CNN & Transfer Learning

Hi! 👋 I'm excited to share my **Emotion Recognition System** Model-2 with you. This project is focused on classifying facial expressions into **three emotions: Happy, Sad, and Angry** using deep learning techniques in Python, powered by **TensorFlow**.

I’ve implemented both a custom **Convolutional Neural Network (CNN)** from scratch and also used **Transfer Learning** with **VGG16** to compare results and leverage pre-trained power. Below, I’ll walk you through the key components of the project and how you can try it out yourself.

---

## 🔍 What This Project Does

This system takes in facial images and identifies the emotion being expressed. It can be particularly useful in:

* Real-time emotion tracking
* Mental health monitoring
* Smart assistants and user experience personalization

The model is trained on a custom dataset of facial expressions organized into three categories and includes techniques like:

* Data Augmentation
* Dropout and Batch Normalization
* Early Stopping
* Learning Rate Scheduling
* Evaluation using F1-Score and Classification Reports

---

## 📁 Dataset Structure

I manually organized the dataset into **training**, **validation**, and **test** sets with the following structure:

```
HASsplittedDS/
├── train/
│   ├── happy/
│   ├── sad/
│   └── angry/
├── val/
│   ├── happy/
│   ├── sad/
│   └── angry/
└── test/
    ├── happy/
    ├── sad/
    └── angry/
```

This setup helps the model generalize better and avoid overfitting.

---

## 🧠 Model Architecture

### CNN from Scratch

I built a CNN model with:

* Multiple convolutional layers
* Batch normalization & dropout for regularization
* Dense layers with L2 regularization
* Categorical softmax output

### Transfer Learning with VGG16

To boost accuracy, I also used **VGG16** (without top layers) and added custom layers on top:

* Frozen base model to retain pre-trained features
* Added Conv2D, Dense, Dropout, and BatchNorm layers
* Tuned using EarlyStopping and ReduceLROnPlateau callbacks

---

## 🚀 How to Run

### Requirements

Make sure you have these installed:

```bash
pip install tensorflow opencv-python matplotlib scikit-learn
```

### Running the Script

Just run the main Python script:

```bash
python emotion_recognition.py
```

The script will either:

* **Load a saved model** (`EmotionRecognitionSystem.h5`) if it exists, or
* **Train a new model** if `train_again = True` is set in the code

All training/validation metrics will be printed and plotted for easy visualization.

---

## 📈 Evaluation

After training, I evaluated the model using:

* Accuracy
* Loss
* Weighted **F1-score**
* Classification report with precision, recall, and support


Output
```
Test Loss: 0.3245
Test Accuracy: 0.9021
Test F1-Score: 0.8967

Classification Report:
              precision    recall  f1-score   support

       angry       0.90      0.89      0.89        45
       happy       0.93      0.94      0.93        50
         sad       0.88      0.88      0.88        40
```

Graphs for training and validation accuracy/loss are also generated using Matplotlib.

---

## 📦 Model Saving

After training, the model is saved as:

```
EmotionRecognitionSystem.h5
```

You can reuse this model without retraining it every time.

---

## 😊 Prediction

The prediction script can be found here: [`Model-2/2-Prediction.py`]([Model-2/2-Prediction.py](https://github.com/bhaarath22/Emotion-RecognitionSystem/blob/35031e082de9d8313aa600b9c4c41b59fd795867/Model-2/2-Prediciton.py))

Now trained deep learning model (`EmotionRecognitionSystem.h5`) is used to **detect and classify human emotions** such as:

* **Happy**
* **Sad**
* **Angry**

It supports two modes of prediction:

### 🎥 Real-Time Webcam Prediction

The system captures frames from your webcam in real time, detects faces, and classifies the emotion of the person in view. For each detected face, it:

1. Detects the face using OpenCV’s Haar Cascade classifier.
2. Preprocesses the face image to match the model’s input format (200x200, normalized).
3. Uses the deep learning model to predict the emotion.
4. Displays the emotion label and a personalized message on the screen.

To run:

```python
predict_emotion(model, use_webcam=True)
```

### 🖼️ Static Image Prediction

You can also use the model to classify emotions from a single image. The process is similar to the webcam mode:

1. Load and preprocess the image.
2. Predict the emotion using the model.
3. Overlay the predicted emotion and an appropriate message on the image.

To run:

```python
predict_emotion(model, use_webcam=False, image_path="path/to/image.jpg")
```
---  
* TensorFlow (Keras) for deep learning inference
* OpenCV for image handling and webcam integration
* NumPy for data processing

---

## 📌 Features

* Real-time emotion detection via webcam
* Offline emotion classification from image files
* Displays personalized messages based on the detected emotion
* Uses Haar Cascade for face detection
* Lightweight and efficient — works on CPU
* Easy-to-extend codebase for additional emotion classes

---
## 📌 Technologies Used

| Library                                   | Purpose                                            |
| ----------------------------------------- | -------------------------------------------------- |
| [TensorFlow](https://www.tensorflow.org/) | Deep learning and model inference                  |
| [OpenCV](https://opencv.org/)             | Image capture, face detection, visualization       |
| [NumPy](https://numpy.org/)               | Image preprocessing and array manipulation         |
| `time`                                    | Timing/logging (used optionally in real-time mode) |

---

## ⚙️ How It Works

### 1. Load Model

```python
model = load_model('EmotionRecognitionSystem.h5')
```

Loads the pre-trained model for emotion classification.

---

### 2. Preprocess Image

```python
frame_resized = cv2.resize(frame, (200, 200))
img_array = tf.keras.preprocessing.image.img_to_array(frame_resized)
img_array /= 255.0
img_array = np.expand_dims(img_array, axis=0)
```

* Resizes to `200x200` (the model input shape)
* Converts to array, normalizes pixels, and adds batch dimension

---

### 3. Face Detection

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
```

* Detects faces in the video/image using Haar cascades

---

### 4. Predict Emotion

```python
prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction)
predicted_class_label = class_labels[predicted_class_index]
```

* Predicts the emotion class from the processed image

---

### 5. Display Results

```python
cv2.putText(frame, f"Emotion: {emotion}", ...)
cv2.putText(frame, message, ...)
```

* Overlays emotion label and a message on the frame

---  
## 🗨️ Emotion Messages Logic

Each detected emotion displays a custom message to provide context and support:

| Emotion | Message                                                                    |
| ------- | -------------------------------------------------------------------------- |
| Happy   | "It's great to see you smiling! Keep spreading those positive vibes."      |
| Sad     | "I see that you’re feeling a bit down. It’s okay to feel sad sometimes..." |
| Angry   | "I notice you might be feeling angry. Take a deep breath..."               |
| Unknown | "Emotion not recognized, but I hope you're feeling okay!"                  |

---

## ✅ Dependencies

Install required Python packages:

```bash
pip install tensorflow opencv-python numpy
```

---  
## 🌱 Future Work

Some improvements I’m considering:

* Adding more emotions (e.g., fear, surprise, disgust)
* Real-time emotion detection using webcam and OpenCV
* Deploying the model using Streamlit or Flask
* Experimenting with other architectures like ResNet, MobileNet, or EfficientNet

---

## 👤 About Me

I’m **Bharath Goud**, passionate about computer vision and deep learning. This project was an exciting way to combine facial recognition with emotion classification.
