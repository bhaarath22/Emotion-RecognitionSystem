# 🎭 ***Emotion Recognition Using Facial Expressions with Scriptural Wisdom Response For psychotherapy***

## ✨ Project Overview

This project presents an innovative **Emotion Recognition System** that classifies human emotions from facial expressions using deep learning, primarily **Convolutional Neural Networks (CNNs)**. It features a user-friendly Graphical User Interface (GUI) that allows real-time emotion detection via webcam or from uploaded images. The unique aspect of this system is its ability to provide **contextual, motivational, and philosophical responses** drawn directly from ancient Hindu scriptures such as the **Vedas, Ramayana, Mahabharata, and Bhagavad Gita**, categorized by the detected emotion.

The system is built on a foundation of iteratively developed CNN models, leveraging advanced techniques to achieve high accuracy in emotion classification. This project seamlessly blends cutting-edge AI with timeless wisdom, offering a unique and insightful user experience.

## 🚀 Features

*   **Real-Time Emotion Detection**: Analyze facial expressions live via your webcam to instantly identify emotions.
*   **Static Image Analysis**: Upload any image containing a face to get an emotion prediction.
*   **8 Emotion Classifications**: The final model is trained to recognize a comprehensive set of 8 emotions: **Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, and Surprise**.
*   **Scriptural Wisdom Integration**: Displays **motivational quotes, shlokas, or teachings** from the Vedas, Ramayana, Mahabharata, and Bhagavad Gita, curated to resonate with the detected emotion [User's Project Summary, 47, 83].
*   **Interactive Graphical User Interface (GUI)**: A intuitive Tkinter-based interface for seamless user interaction, including live camera feed display and result presentation.
*   **Robust Face Detection**: Utilizes **OpenCV's Haar Cascade Classifiers** for accurate and efficient face localization within images or video streams.
*   **Model Performance Feedback**: Provides **prediction accuracy** and a clear display of the classified emotion.
*   **Responsive Design**: The GUI is configured for a fullscreen layout that adapts to screen dimensions, enhancing user experience.
*   **Error Handling & Demo Mode**: Gracefully handles scenarios where the pre-trained model is not found, running in a "demo mode" as a fallback.

## 🛠️ Technologies Used

This project harnesses a powerful stack of Python libraries and deep learning frameworks:

*   **Deep Learning Framework**:
    *   **TensorFlow / Keras**: The core framework for building, training, and deploying CNN models.
        *   **Keras Layers**: Utilizes various layers like `Conv2D`, `MaxPooling2D`, `Dense`, `BatchNormalization`, `Dropout`, `Resizing`, `Rescaling` for model architecture and preprocessing.
        *   **Keras Callbacks**: Implements `EarlyStopping` and `ModelCheckpoint` for efficient and robust model training.
*   **Computer Vision**:
    *   **OpenCV (`cv2`)**: Essential for real-time video capture, image processing (resizing, color conversion), and face detection using Haar cascades.
*   **Graphical User Interface**:
    *   **Tkinter**: Python's standard GUI toolkit used to create the interactive interface.
    *   **Pillow (`PIL`)**: Used for image processing and displaying images within the Tkinter GUI.
*   **Numerical Operations**:
    *   **NumPy**: Fundamental for efficient numerical operations, array manipulation, and preprocessing image data.
*   **Data Handling**:
    *   **`os`**: For file path operations and interacting with the operating system.
    *   **`json`**: For loading the emotion-specific messages from a JSON file.
*   **Model Evaluation & Visualization**:
    *   **Matplotlib**: Used for plotting training history, accuracy, and loss curves, as well as confusion matrices.
    *   **Scikit-learn**: Provides tools for advanced model evaluation metrics such as **F1-score**, **classification reports** (precision, recall, support), and **confusion matrices**.
    *   **TensorFlow Probability (`tfp`)**: Used specifically for the **CutMix** data augmentation technique.

## 📈 Model Development Journey

The project evolved through several iterations of CNN model development, with each step incorporating more advanced techniques to enhance accuracy and robustness.

### **Model 1: Initial Exploration (Happy vs. NotHappy)**

*   **Goal**: To build a foundational CNN for binary emotion classification: `happy` or `nothappy`.
*   **Architecture**: A basic CNN comprising multiple `Conv2D` layers with ReLU activation, `MaxPooling`, `Flatten`, `Dense` layers, and a `Dropout` layer before a `Softmax` output.
*   **Challenges & Learnings**:
    *   **Overfitting**: Noted a significant gap between training and validation accuracy, likely due to a small dataset.
    *   **Data Imbalance**: Identified potential unevenness in the number of images per class.
    *   **Limited Variation**: Recognized the need for more diverse facial images.
*   **Planned Improvements (Paving the way for future models)**: Data augmentation, increasing dataset size, early stopping, regularization, and exploring transfer learning.

### **Model 2: Expanding Emotions & Introducing Transfer Learning (Happy, Sad, Angry)**

*   **Goal**: Expanded classification to three emotions: `Happy`, `Sad`, and `Angry`.
*   **Key Techniques Implemented**:
    *   **Custom CNN from Scratch**: Built a more sophisticated custom CNN with `Batch Normalization` and `L2 regularization` in dense layers.
    *   **Transfer Learning with VGG16**: Integrated **VGG16** (a powerful pre-trained CNN) as a base model (with its top layers removed and frozen) to leverage its learned features, adding custom layers on top for fine-tuning.
    *   **Advanced Training Aids**: Incorporated `EarlyStopping` and `ReduceLROnPlateau` callbacks for more controlled and efficient training.
    *   **Comprehensive Evaluation**: Evaluated using **weighted F1-score** and **classification reports** in addition to accuracy and loss.
*   **Results**: Achieved a significantly improved performance with a **Test Accuracy of approximately 90.21%** and a Test F1-Score of 89.67%.

### **Model 3: Final Robust System (8 Emotions with LeNet-Inspired Architecture)**

*   **Goal**: Classified **8 distinct emotions**: `Anger`, `Contempt`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, and `Surprise`.
*   **Architecture**: A **LeNet-Inspired CNN** architecture, structured with preprocessing layers (`Resizing`, `Rescaling`), multiple convolutional blocks (including `Conv2D`, `BatchNormalization`, `MaxPool2D`, `Dropout`), `Flatten` layer, and two `Dense` layers, culminating in a `Softmax` output layer for multi-class probability prediction.
*   **Advanced Data Augmentation**: Implemented a comprehensive data augmentation pipeline including:
    *   **Built-in Keras Layers**: `RandomRotation`, `RandomFlip`, `RandomContrast`, `RandomBrightness`, `RandomTranslation`.
    *   **Custom Augmentations**: `AddGaussianNoise` and `ColorJitter` for more diverse training data.
    *   **CutMix**: A powerful technique that blends patches and labels from two images, significantly enhancing generalization and preventing overfitting.
*   **Configuration Management**: All important hyperparameters are managed through a centralized `Configuration` dictionary, making the model highly customizable and reproducible.
*   **Rigorous Evaluation**: The final model is thoroughly evaluated on a dedicated `TestingDS` dataset, with detailed metrics like **loss, accuracy, top-k accuracy, weighted F1-score, and a confusion matrix** to provide a holistic view of performance.

## 📁 Dataset

The models were trained on a **custom dataset of facial expressions**, carefully organized to facilitate deep learning training. The dataset follows a standard folder-based structure, ideal for use with `tf.keras.utils.image_dataset_from_directory`:

*   **Image Preprocessing**: All images are automatically resized to **256x256 pixels** and normalized to a `` range for consistent model input.
*   **Automated Labeling**: Images are automatically labeled based on their respective directory names, and these labels are converted into **one-hot encoded categorical vectors**.
*   **Extensive Augmentation**: The dataset is augmented extensively using a variety of techniques (as detailed in Model 3's section) to increase its diversity and prevent overfitting, thereby improving model generalization.

## 📜 Scriptural Wisdom Integration

A distinctive feature of this project is the integration of profound wisdom from Hindu scriptures. The contextual messages are stored in a **JSON file (e.g., `LinesForEmotions.json`)**, structured for easy access and expansion:

```json
{
  "happy": [
    "Keep smiling! Your joy is contagious!",
    "\"Sukham eva hi duhkhaanam antyam\" - True happiness lies beyond the fleeting nature of sorrow. (Bhagavad Gita)"
  ],
  "sad": [
    "It's okay to feel sad. Better days are ahead.",
    "\"Sarvam duhkham duhkham\" - All is suffering, all is sorrow. Recognizing this is the first step towards liberation. (Vedas)"
  ],
  "angry": [
    "I notice you might be feeling angry. Take a deep breath...",
    "\"Krodhaadbhavati sammohah\" - From anger comes delusion, from delusion loss of memory, from loss of memory the ruin of intelligence, and from the ruin of intelligence, one perishes. (Bhagavad Gita)"
  ],
  // ... and so on for all 8 emotions
}
```

When an emotion is detected, the system randomly selects and displays one of these pre-categorized messages, providing a unique blend of modern AI and ancient philosophical guidance. This adds a deep, reflective, and supportive dimension to the user's interaction with the system.

## ⚙️ How to Run Locally

Follow these steps to set up and run the Emotion Recognition System on your local machine.

### **1. Prerequisites**

Ensure you have Python installed (Python 3.8+ is recommended).

### **2. Clone the Repository**

First, clone this GitHub repository to your local machine:

```bash
git clone https://github.com/your-username/Emotion-Recognition-with-Scriptural-Wisdom.git
cd Emotion-Recognition-with-Scriptural-Wisdom
```

**(Note: Replace `https://github.com/your-username/Emotion-Recognition-with-Scriptural-Wisdom.git` with your actual repository URL.)**

### **3. Install Dependencies**

Install the required Python packages using pip. It's recommended to do this within a virtual environment.

```bash
pip install tensorflow opencv-python numpy
```

If you encounter issues with webcam display or OpenCV window not opening, you might need to install `opencv-python-headless`:

```bash
pip install opencv-python-headless
```

### **4. Model & Data Setup**

*   **Pre-trained Model**: The system expects a pre-trained model file named `EmotionRecognitionSystemFinal1.h5` in the appropriate directory (e.g., `/Main/ERS/` as suggested by the source). If this file is not found, the system will run in "demo mode".
    *   **Recommendation**: Place your trained `EmotionRecognitionSystemFinal1.h5` model file in the expected path or adjust the `model_path` variable in your `3-Interface.py` (or main GUI script) accordingly.
*   **Emotion Messages JSON**: Ensure your `LinesForEmotions.json` file is present in the expected location for the scriptural wisdom responses.

### **5. Run the Application**

Navigate to the directory containing your main GUI script (e.g., `3-Interface.py` or `EmotionRecognitionGUI.py`) and run it:

```bash
python 3-Interface.py
```

This will launch the **Graphical User Interface (GUI)**.

### **6. Usage within the GUI**

Once the GUI opens:

*   **Start Webcam**: Click the **"Start Webcam"** button to activate your live camera feed. The system will then detect faces in real-time and display emotions along with scriptural messages.
*   **Upload Image**: Click the **"Upload Image"** button to select an image file from your computer. The system will process the image, detect faces, and show the predicted emotion and message.
*   **Stop**: Click the **"Stop"** button to stop the webcam feed and clear the display.

#### **(Alternative: Command-Line Prediction)**

You can also run a separate prediction script if available (e.g., `2-Prediction.py`) for command-line interaction.

```bash
python 2-Prediction.py
```

Follow the prompts:
*   Enter `'webcam'` to use the webcam feed (press `q` to quit).
*   Enter `'image'` and then provide the full path to an image file.

## 📸 Screenshots

*(To be updated with actual screenshots of the GUI in action, showing both webcam and image upload modes with emotion detection and scriptural messages.)*

![Screenshot 1: Main GUI with Webcam Feed](placeholder_webcam.png)
*Figure 1: Real-time emotion detection via webcam with scriptural wisdom response.*

![Screenshot 2: GUI with Uploaded Image Analysis](placeholder_image.png)
*Figure 2: Emotion analysis from an uploaded image.*

## 🌱 Future Improvements

I am continuously working to enhance this project. Here are some planned future developments:

*   **Expand Emotion Classes**: Integrate detection for more nuanced emotions (e.g., `fear`, `surprise`, `disgust` beyond the current 8, if applicable, or even more granularity).
*   **Alternative Model Architectures**: Experiment with other state-of-the-art CNN architectures such as **ResNet, MobileNet, or EfficientNet** for potential performance gains and efficiency improvements.
*   **Model Deployment**: Explore options for deploying the model as a web application using frameworks like **Streamlit or Flask** for broader accessibility.
*   **Dataset Expansion**: Continue to increase the diversity and size of the training dataset to further improve generalization.
*   **Advanced Regularization Techniques**: Investigate and implement additional regularization methods to combat overfitting.

## 🙏 Credits & Acknowledgments

*   **Scriptural Sources**: The profound and inspiring wisdom shared in this project is humbly drawn from the sacred texts of **The Vedas, The Ramayana, The Mahabharata, and The Bhagavad Gita**. Their timeless teachings provide immense value and guidance.
*   **Deep Learning & Computer Vision Libraries**: Special thanks to the developers and communities behind **TensorFlow, Keras, OpenCV, NumPy, Scikit-learn, Matplotlib, Pillow, and Tkinter** for providing the powerful tools that made this project possible.

---
```
