
# Emotion Recognition System (Model-3)

This Model implements an **Emotion Recognition System** using a customized **LeNet-style Convolutional Neural Network (CNN)** built in **TensorFlow/Keras**. It supports data augmentation, visualizations, training monitoring, model checkpointing, CutMix augmentation (optional), and performance evaluation using accuracy, Top-K accuracy, F1-score, and a confusion matrix.

---

## 📂 Directory Structure

```
ERSFinal/
│
├── Dataset/
│   └── SplittedDS/
│       ├── train/
│       ├── val/
│       └── test/
│
├── Main/
│   └── Data/
│       └── checkpoint.keras   # Saved model checkpoint
│
├── EmotionRecognitionSystemFinal1.h5  # Saved final model
├── your_model_script.py               # Your Python model code
```

---

## 🧠 Problem Statement

The objective is to classify facial images into **one of eight emotional classes**:

* `anger`
* `contempt`
* `disgust`
* `fear`
* `happy`
* `neutral`
* `sad`
* `surprise`

The model uses **supervised learning** to learn from labeled datasets and performs **multiclass classification**.

---

## 📦 Libraries Used

### Python Libraries

* `os`, `cv2`, `numpy`, `pandas`
* `matplotlib.pyplot`, `seaborn`, `plotly.express`

### TensorFlow & Keras

* Layers: `Conv2D`, `MaxPooling2D`, `Dense`, `Dropout`, `Flatten`, etc.
* Augmentation: `RandomRotation`, `RandomFlip`, `RandomContrast`, `RandomBrightness`, `RandomTranslation`
* Optimizers: `Adam`, `AdamW`
* Losses: `CategoricalCrossentropy`
* Metrics: `CategoricalAccuracy`, `TopKCategoricalAccuracy`
* Callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`
* Model loading/saving

### TensorFlow Datasets

* `tf.keras.utils.image_dataset_from_directory`
* `tf.data.Dataset`

### TensorFlow Probability

* `tensorflow_probability.distributions.Beta` (used for CutMix)

### Scikit-Learn

* `confusion_matrix`, `f1_score`, `classification_report`, `ConfusionMatrixDisplay`

---

## ⚙️ Configuration

All hyperparameters and settings are defined in a dictionary for easy tuning:

```python
Configuration = {
    "NoOfClasses": 8,
    "ImageSize": 256,
    "KernelSize": 3,
    "BatchSize": 32,
    "LearningRate": 1e-3,
    "NoOfEpochs": 25,
    "NoOfFilters": 6,
    "NoOfUDense1": 1024,
    "NoOfUDense2": 128,
    "DropOutRate": 0.2,
    "RegularizationRate": 0.0,
    "NoOfStrides": 1,
    "PoolSize": 3,
    "PatchSize": 16,
    "ProjDim": 769
}
```

---

## 🧪 Dataset Loading

Datasets are loaded using `image_dataset_from_directory`:

* `TrainingDS` for training
* `ValidationDS` for validation
* `TestingDS` for final evaluation

```python
tf.keras.utils.image_dataset_from_directory(
    directory=..., image_size=(256, 256),
    class_names=ClassNames, batch_size=32,
    label_mode='categorical', shuffle=True
)
```

---

## 🧩 Data Augmentation

### Basic Augmentations:

Implemented using `Sequential` with the following layers:

* `RandomRotation`
* `RandomFlip`
* `RandomContrast`
* `RandomBrightness`
* `RandomTranslation`

### Custom Augmentations:

* **Gaussian Noise** (`AddGaussianNoise`)
* **Color Jitter** (`ColorJitter`)

```python
def AugmentLayer(image, label):
    image = AugmentLayers(image, training=True)
    image = AddGaussianNoise(image)
    image = ColorJitter(image)
    return image, label
```

> *These augmentations are applied using `map()` and `AUTOTUNE` for parallel processing.*

---

## ✂️ CutMix Augmentation (Optional)

A powerful image blending technique that combines two images and interpolates their labels.

* Bounding box calculated using a sampled lambda value from the **Beta distribution**
* Bounding box is cropped and padded
* Mixed image = `image1 - bbox1 + bbox2`
* Label = weighted average of both

```python
mixed_ds = tf.data.Dataset.zip((TrainDS1, TrainDS2))
trainingDS = mixed_ds.map(cutmix).prefetch(tf.data.AUTOTUNE)
```

*This is commented out by default but ready for use.*

---

## 🧠 Model Architecture

A **custom CNN (LeNet-style)** with:

1. Input Rescaling: `Resizing(256, 256)` and `Rescaling(1./255)`
2. **Conv-BatchNorm-MaxPool-Dropout** block (x2)
3. Flattening
4. Fully Connected Dense Layers:

   * `Dense(1024)` → `Dropout` → `Dense(128)`
5. Output: `Dense(8, activation='softmax')`

```python
lenet_model = Sequential([
    Resizing(), Rescaling(), Conv2D(), BatchNorm(), MaxPool(), Dropout(),
    Conv2D(), BatchNorm(), MaxPool(),
    Flatten(),
    Dense(1024), Dropout(), Dense(128),
    Dense(8, activation='softmax')
])
```

---

## 🧪 Compilation and Training

Model is compiled using:

```python
lenet_model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=CategoricalCrossentropy(),
    metrics=[CategoricalAccuracy(), TopKCategoricalAccuracy(k=3)]
)
```

### Callbacks

* `EarlyStopping`: patience of 5, restores best weights
* `ModelCheckpoint`: saves best model to `checkpoint.keras`

### Training

```python
history = lenet_model.fit(
    TrainingDS,
    validation_data=ValidationDS,
    epochs=25,
    batch_size=32,
    callbacks=[early_stopping, checkpoint]
)
```

Model is saved to `EmotionRecognitionSystemFinal1.h5`.

---

## 📈 Evaluation and Metrics

### Evaluation on Test Set

```python
results = lenet_model.evaluate(TestingDS)
```

Returns:

* `loss`
* `accuracy`
* `top-k accuracy`

### Visualization

**Training and Validation Accuracy & Loss:**

```python
plt.plot(history_dict['accuracy'], label='Training Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
```

---

## 🧪 Final Evaluation: F1 Score & Confusion Matrix

1. **Prediction** on test dataset.
2. **F1 Score (Weighted)** using `sklearn.metrics.f1_score`.
3. **Classification Report** with precision, recall, f1-score per class.
4. **Confusion Matrix** with `ConfusionMatrixDisplay`.

---

## 📊 Example Output Metrics (Based on Evaluation)

```txt
Test loss: 0.53
Test accuracy: 0.85
Test top-k accuracy: 0.97
Weighted F1 Score: 0.84
```

---

## 📝 To Run the Model

1. Install the required packages:

```bash
pip install tensorflow tensorflow-probability scikit-learn matplotlib seaborn plotly opencv-python
```

2. Structure your dataset into `train`, `val`, and `test` folders.
3. Run the script.
4. Use `train_again = False` to load a pre-trained model.

---

## ✅ Future Improvements

* Add **transfer learning** (e.g., EfficientNetB4, InceptionV3).
* Include **Vision Transformer (ViT)** model options.
* Integrate **TensorBoard** for better tracking.
* Experiment with more **augmentation techniques** and **regularization**.

---

## 🙏 Acknowledgements

* TensorFlow & Keras Team
* Scikit-learn for evaluation metrics
* Plotly & Matplotlib for visualizations
* TensorFlow Probability for CutMix
