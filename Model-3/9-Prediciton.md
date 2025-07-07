## 🧠 Emotions Detected
trained model supports classification of the following 8 emotions: 

* 😠 Anger
* 😒 Contempt
* 🤢 Disgust
* 😨 Fear
* 😄 Happy
* 😐 Neutral
* 😢 Sad
* 😲 Surprise

---  
## 🔧 Requirements

```bash
pip install tensorflow opencv-python numpy
```

issues with the webcam display or CV2 window not opening

```bash
pip install opencv-python-headless
```

---

## 🏗️ How It Works

### 1. **Model Loading**

The script loads trained CNN model (`EmotionRecognitionSystemFinal1.h5`). If the model file is not found, the system runs in **demo mode** using a placeholder function.

### 2. **Face Detection**

OpenCV's Haar cascade is used to detect faces in the image or webcam feed:

```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
```

Only the **first detected face** is used for emotion prediction.

### 3. **Emotion Prediction**

Detected face regions are resized and normalized before being passed into the model:

```python
face = cv2.resize(face, (256, 256))
face = face / 255.0
face = np.expand_dims(face, axis=0)
```

The model returns a prediction vector, which is mapped to one of the emotion labels using `argmax()`.

### 4. **Custom Emotion Messages**

A JSON file (`LinesForEmotions.json`) contains a list of motivational or context-specific lines per emotion.

Example:

```json
{
  "happy": ["Keep smiling!", "Your joy is contagious!"],
  "sad": ["It's okay to feel sad. Better days are ahead."]
}
```

One message is randomly selected and printed after the emotion is predicted.

### 5. **Modes of Operation**

* **Image Mode**: You provide the path to an image, and the system prints the predicted emotion and message.

* **Webcam Mode**: The system continuously reads from the webcam, detects emotions in real-time, and prints messages.

---

## 🖥️ Usage

Run the Prediction script 

then we get this message on the screen:

```
Enter 'webcam' to use the webcam or 'image' to upload an image:
```

### 💻 Webcam Mode

```text
Enter 'webcam'
```

* Starts your webcam.
* Detects your facial emotion in real time.
* Press `q` to quit.

### 🖼️ Image Mode

```text
Enter 'image'
Enter the path to the image: /path/to/image.jpg
```

* Detects and predicts the emotion from the provided image.

---

```text
Emotion: happy, Accuracy: 97.12%
Emotion Message: Your smile brightens the day!
```

If no face is detected:

```text
No face detected in the image.
```

---  
