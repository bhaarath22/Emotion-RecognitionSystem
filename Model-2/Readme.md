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

## 🧠 Model Overview

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

## 🌱 Future Work

Some improvements I’m considering:

* Adding more emotions (e.g., fear, surprise, disgust)
* Real-time emotion detection using webcam and OpenCV
* Deploying the model using Streamlit or Flask
* Experimenting with other architectures like ResNet, MobileNet, or EfficientNet

---

## 👤 About Me

I’m **Bharath Goud**, passionate about computer vision and deep learning. This project was an exciting way to combine facial recognition with emotion classification.
