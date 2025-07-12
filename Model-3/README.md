# ❤️ Emotion Recognition System (Transfer learning , Data Augmentation) Model-3

 **Model-3**! 🎉 This Model is designed to accurately identify 8-human emotions, such as happiness, sadness, anger, and more, from facial expressions. It leverages advanced deep learning techniques and a custom-built GUI to provide a seamless and interactive experience.
  
## 🚀 Features

*   **Real-Time Emotion Detection**: Instantly detect emotions from your webcam feed, providing a live and dynamic experience.
*   **Image Upload & Analysis**: Upload any image containing a face, and the system will process it to identify the dominant emotion.
*   **Visual Feedback**: See a **bounding box** around detected faces, along with the predicted **emotion label** and its **accuracy percentage** directly on the image/feed.
*   **Interactive GUI**: A full-screen, intuitive interface built with Tkinter for easy interaction.
*   **Informative Messages**: Based on the detected emotion, the system provides helpful and context-aware messages, loaded dynamically from a JSON file.
*   **Core Functionality (CLI)**: For those who prefer the command line, there's also a console-based interface to process images or run the webcam feed.
*   **Robust Deep Learning Model**: At its heart, a carefully trained deep learning model ensures high accuracy in emotion classification.
---

## 🧠 How It Works

1.  **Face Detection** 🔍:
    *   The first step is to locate human faces within an image or video frame. I use **OpenCV's Haar Cascade Classifiers**, specifically `haarcascade_frontalface_default.xml`, which is highly effective for detecting frontal faces.
    *   Once detected, the face region is isolated for further processing.

2.  **Emotion Prediction** 🤔:
    *   The isolated face image is then fed into my **pre-trained deep learning model** (`EmotionRecognitionSystemFinal1.h5`).
    *   Before prediction, the face image is **resized to 256x256 pixels** and its pixel values are **normalized** (scaled between 0 and 1). This preprocessing step is crucial for the model's performance.
    *   The model then outputs a prediction, which is mapped to one of the **8 supported emotion labels**: `['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']`.
    *   Along with the emotion label, the system also calculates and displays the **confidence (accuracy)** of the prediction.

3.  **Dynamic Messaging** 💬:
    *   To make the experience more engaging, the system retrieves a random, pre-defined message corresponding to the detected emotion. These messages are stored in a JSON file (`LinesForEmotions.json`).
    *   This provides a touch of 'wisdom' or relevant commentary alongside the technical output.
---


## 🛠️ Technologies Used

This project is built using a powerful stack of Python libraries:

*   **Python**: The core programming language.
*   **TensorFlow & Keras**: My go-to framework for building, training, and deploying deep learning models.
*   **OpenCV (`cv2`)**: Essential for all image and video processing tasks, including face detection and drawing annotations.
*   **Tkinter**: Python's standard GUI toolkit, used to create the interactive and attractive user interface.
*   **NumPy**: Fundamental for numerical operations, especially array manipulation for image data.
*   **PIL (Pillow)**: Used for handling and converting image formats for the Tkinter GUI.
*   **JSON**: For storing and retrieving the emotion-specific messages.
---


## 📈 Model Development Journey (LeNet-like Architecture)

My model development focused on building a robust convolutional neural network (CNN) inspired by the classic **LeNet architecture**, tailored for emotion recognition. Here's a deep dive:

*   **Architecture Overview** 🏗️:
    The model is a `Sequential` Keras model, designed to process images of size 256x256. It begins with `Resizing` and `Rescaling` layers to ensure input consistency. The core structure consists of:
    *   **Convolutional Blocks**: Multiple `Conv2D` layers (with `filters` such as 6 initially, then 16) extract features, followed by `BatchNormalization` for stable training, `MaxPool2D` for down-sampling, and `Dropout` layers to prevent overfitting.
    *   **Flattening**: After convolutional layers, the feature maps are `Flatten`ed into a single vector.
    *   **Dense Layers**: This is followed by two fully connected (`Dense`) layers (with 1024 and 128 units respectively), also incorporating `BatchNormalization` and `Dropout`.
    *   **Output Layer**: The final `Dense` layer has 8 units (one for each emotion class) with a `softmax` activation function, outputting probability distributions over the emotions.

*   **Key Configuration Parameters** ⚙️ (from my `Configuration` dictionary):
    *   `NoOfClasses`: 8 (for the 8 emotions)
    *   `ImageSize`: 256x256
    *   `KernelSize`: 3 (for convolutional filters)
    *   `BatchSize`: 32
    *   `LearningRate`: 1e-3 (for the Adam optimizer)
    *   `NoOfEpochs`: 25 (maximum training epochs)
    *   `NoOfFilters`: 6 (initial filters in Conv2D layers, scaling up later)
    *   `NoOfUDense1`: 1024 (units in the first dense layer)
    *   `NoOfUDense2`: 128 (units in the second dense layer)
    *   `DropOutRate`: 0.2 (for regularization)
    *   `PoolSize`: 3 (for max-pooling layers)
---  
*   **Advanced Data Augmentation** 🎨:
    To make the model more robust and generalize better to unseen data, I implemented several data augmentation techniques:
    *   **Basic Augmentations**: Random `Rotation` (up to 2.5 degrees), horizontal `Flip`, `Contrast` (up to 10%), `Brightness` (10-20%), and `Translation` (up to 10% height/width).
    *   **Custom Augmentations**: Added `AddGaussianNoise` to introduce random noise, and `ColorJitter` to randomly adjust brightness and contrast.
    *   **CutMix**: A powerful technique where a patch from one image is cut and pasted onto another image, and the labels are mixed proportionally. This significantly improves the model's ability to learn from diverse features.
---

*   **Training & Evaluation** 📊:
    *   **Compilation**: The model was compiled using the `Adam` optimizer with a learning rate of 1e-3, `CategoricalCrossentropy` as the loss function, and `CategoricalAccuracy` and `TopKCategoricalAccuracy(k=3)` as metrics.
    *   **Callbacks**: To ensure optimal training, I used:
        *   `EarlyStopping`: Stops training if `val_loss` doesn't improve for 5 consecutive epochs, restoring the best weights.
        *   `ModelCheckpoint`: Saves the best performing model (based on `val_loss`) during training to `checkpoint.keras` and finally to `EmotionRecognitionSystemFinal1.h5`.
    *   **Performance**: After training for up to 25 epochs, the model was evaluated on the `TestingDS`. The results, including **Test loss**, **Test accuracy**, and **Test top-k accuracy**, were printed.
    *   **Visualizations & Metrics**: Post-training analysis included plots of **Training and Validation Accuracy/Loss**, and a comprehensive evaluation with **Weighted F1 Score**, a **Classification Report**, and a detailed **Confusion Matrix**. These metrics are crucial for understanding how well the model performs across all emotion classes.

---

## 🧪 GUI

Once the GUI application is running:

1.  **Start Webcam** 📹:
    *   Click the **"Start Webcam"** button located on the left.
    *   Your webcam feed will appear in the central display area.
    *   The system will continuously attempt to detect faces and predict emotions from the live feed. (Note: The current GUI version for webcam primarily displays the feed. For continuous bounding box and emotion display on live feed, further integration of detection within the `update_frame` loop would be a great future improvement!)

2.  **Upload Image** 📸:
    *   Click the **"Upload Image"** button.
    *   A file dialog will open, allowing you to select an image file from your computer.
    *   Once an image is selected, the system will process it, detect faces, predict emotions, and display the result with a bounding box and text.

3.  **Stop Operation** 🛑:
    *   Click the **"Stop"** button to cease the webcam feed or clear the currently displayed image. This will also reset the emotion label.

4.  **View Results** 📝:
    *   The detected emotion, its accuracy, and a corresponding informative message will be displayed in the **"Accuracy&loss System"** message box on the top right of the screen.
---
![Click](Model-3/Data/GUI)
## 🌱 Future Improvements

I'm always looking to enhance this project! Here are some ideas for future development:


*   **More Diverse Datasets**: Expand the training dataset with more diverse faces, lighting conditions, and expressions to improve generalization.
*   **Explore Advanced Architectures**: Experiment with cutting-edge deep learning models like **EfficientNetB4, VGG16, ResNet50, or InceptionV3** (which are already imported in the training script!) for potentially higher accuracy and efficiency.
*   **Emotion Intensity**: Develop the model to not just classify emotions but also predict their intensity or detect more subtle emotional nuances.
*   **Improved UI/UX**: Enhance the graphical interface with more advanced controls, interactive visualizations, and a more polished design.
*   **Cross-Platform Deployment**: Package the application for easier distribution across different operating systems.
*   **Performance Optimization**: Optimize the model for faster inference on various hardware, including potentially mobile or edge devices.
*   **Integration with APIs**: Explore possibilities of integrating this system with other applications or web services.
