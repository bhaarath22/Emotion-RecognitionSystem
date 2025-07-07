
# **Emotion Recognition Using CNN**   ***Model-1***

Hi, This is a  project I worked on to recognize human emotions from facial expressions using Convolutional Neural Networks (CNNs). emotion recognition system that classifies facial expressions into two categories: **happy** and **not happy**. It uses CNNs with TensorFlow/Keras and OpenCV to build and train a deep learning model from scratch.

---

## 📁 Dataset

The dataset is structured into three parts:

- `training/` — used to train the model
- `validation/` — used to validate the model during training
- `testing/` — used to evaluate the final model

Each folder contains two subfolders:
- `happy/` — images with happy expressions
- `nothappy/` — images with neutral, sad, angry, or other non-happy expressions

All images are resized to 256x256 pixels and normalized to a [0,1] range.

---

## 📦 Libraries Used

- TensorFlow / Keras
- OpenCV (cv2)
- NumPy
- Matplotlib
- Scikit-learn

---

## 🏗️ ***Model Architecture***

The CNN architecture used:

```text
Input Layer: 256x256x3 images
Conv2D(32 filters) + ReLU → MaxPooling(2x2)
Conv2D(64 filters) + ReLU → MaxPooling(2x2)
Conv2D(128 filters) + ReLU → MaxPooling(2x2)
Flatten
Dense(128) + ReLU
Dropout(0.5)
Output Layer: Dense(2) + Softmax
```


Loss Function: `categorical_crossentropy`  
Optimizer: `adam`  
Metrics: `accuracy`

---

## 📊 ***Training***

The model is trained for 30 epochs with a batch size of 32.

If a model already exists (`EMR_NDA.h5`), it loads the saved model instead of training again.
During training, accuracy and loss are plotted for both training and validation sets.

---

## ✅ ***Evaluation***

- Final test accuracy: **~68.18%**
- Confusion matrix is plotted to evaluate classification performance.
- Additional plots show loss and accuracy per epoch.

---
# **Results**  
## Accuracy and Loss  
![img}(Model-1/Data/Accuracy&loss/A&C.png)  
---  
## Confusion Matrix
![Cimg](Model-1/Data/Accuracy&loss/Confusion.png)  
## Training 
![img3](Model-1/Data/Accuracy&loss/Training.png)
## 🔍 ***Observations & Challenges***

1. **Overfitting Detected**
   - Training accuracy is much higher than validation/test accuracy.
   - Likely due to the **small dataset size**.

2. **Data Imbalance**
   - Number of images in `happy` and `nothappy` categories might be uneven.

3. **Limited Variation**
   - Facial images may lack diversity (age, ethnicity, lighting, pose).

---
## 🧠 ***Planned Improvements***

1. **Data Augmentation**
   - Use Keras' `ImageDataGenerator` to:
     - Flip, rotate, zoom, and shift images
     - Add noise, change brightness
   - Increases variety and prevents overfitting.

2. **Increase Dataset Size**
   - Add more images or use public datasets (FER2013, AffectNet).

3. **Early Stopping & Checkpoints**
   - Stop training when validation accuracy stops improving.
   - Save best model during training.

4. **Regularization**
   - Add L2 regularization to convolutional and dense layers.

5. **Transfer Learning**
   - Use pre-trained CNNs like VGG16, ResNet50 with fine-tuning.

---

## 💬 Final Thoughts

This project helped me explore:
- Building CNNs from scratch
- Loading and preprocessing image data
- Visualizing model performance
- Identifying overfitting and mitigation strategies

It's a great foundation for building more advanced emotion recognition systems.

---
