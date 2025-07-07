## 📦 Python Imports 
---

### 1️⃣ File Handling and Environment Setup

These libraries help manage file structures, datasets, and system-level interactions.

```python
import os
```

- `os`: Enables interaction with the operating system. Essential for:
  - Navigating directories (`os.listdir`, `os.path`)
  - Creating folders
  - Dynamically loading data

---

### 2️⃣ Image Processing

```python
import cv2
```

- `cv2` (OpenCV): A powerful computer vision library used for:
  - Reading and writing images
  - Image augmentation
  - Real-time image transformations (resizing, color conversion, etc.)

---

### 3️⃣ Numerical & Data Manipulation

```python
import numpy as np
import pandas as pd
```

- `numpy`: The backbone of numerical computing in Python.
  - Efficient matrix operations (used for tensors, image arrays)
  - Essential for preprocessing and custom logic in ML pipelines

- `pandas`: Useful for data analysis and managing structured data (e.g., CSVs).
  - Ideal for reading labels, annotations, and organizing metadata

---

### 4️⃣ Data Visualization & Exploration

```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
```

- `matplotlib.pyplot`: Core plotting library (static plots)
  - Used for plotting accuracy/loss curves, histograms

- `seaborn`: High-level API for statistical graphics
  - Great for heatmaps, confusion matrices, KDE plots

- `plotly.express`: Interactive plotting
  - Useful for dynamic visual analysis, especially with larger datasets

---

### 5️⃣ TensorFlow & Keras Core API

```python
import tensorflow as tf
from tensorflow.keras import Sequential, Model
```

- `tensorflow`: Deep learning framework used for:
  - Building, training, and evaluating neural networks
- `Sequential`: A linear stack of layers for simple models
- `Model`: Subclassing API for more flexible architectures

---

### 6️⃣ Keras Layers — Building Blocks of CNNs

```python
from tensorflow.keras.layers import (
    Resizing, Rescaling, RandomRotation, RandomFlip, RandomContrast, RandomBrightness, RandomTranslation,
    Layer, Input, InputLayer, Add,
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Flatten, Dense, BatchNormalization, Dropout, Activation
)
```

📚 Layer Categories:

- 🧩 Input & Preprocessing:
  - `Input`, `InputLayer`, `Resizing`, `Rescaling`: Normalize input images, resize on-the-fly

- 🧪 Data Augmentation:
  - `RandomRotation`, `RandomFlip`, `RandomTranslation`, `RandomBrightness`, `RandomContrast`: Improves generalization

- 🔍 Core CNN Layers:
  - `Conv2D`: Extracts features using filters/kernels
  - `MaxPooling2D`: Reduces spatial dimensions while preserving key features
  - `GlobalAveragePooling2D`: Converts feature maps into a vector by averaging spatial dimensions

- 🎛️ Dense Layers & Output:
  - `Flatten`, `Dense`, `Activation`: Create fully connected layers and final classifier
  - `Dropout`: Prevents overfitting by randomly disabling neurons
  - `BatchNormalization`: Stabilizes and speeds up training

---

### 7️⃣ Loss Functions — Measuring Model Error

```python
from tensorflow.keras.losses import CategoricalCrossentropy
```

- `CategoricalCrossentropy`: Used for multi-class classification where labels are one-hot encoded

🧠 Alternatives:
- `SparseCategoricalCrossentropy`: Use when labels are integer-encoded
- `BinaryCrossentropy`: Binary classification tasks
- `KL Divergence`, `Mean Squared Error (MSE)`: For custom or regression tasks

---

### 8️⃣ Regularization

```python
from tensorflow.keras.regularizers import l2
```

- `l2`: Adds a penalty to large weights (L2 regularization), discouraging overfitting

🔧 Other techniques:
- `L1`: Adds absolute values of weights
- `Elastic Net`: Combines L1 and L2
- `Dropout` and `BatchNormalization` also serve as regularization

---

### 9️⃣ Evaluation Metrics

```python
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
```

- `CategoricalAccuracy`: Measures if predicted label = true label
- `TopKCategoricalAccuracy`: Checks if the correct class is within top-K predictions

🎯 Other useful metrics:
- `Precision`, `Recall`, `AUC`
- `Sparse` versions for non-one-hot labels

---

### 🔁 Callbacks — Dynamic Training Control

```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
```

- `ModelCheckpoint`: Saves model weights at best performance (e.g., lowest val_loss)
- `EarlyStopping`: Stops training if validation performance plateaus
- `ReduceLROnPlateau`: Automatically reduces learning rate on plateaus

🚦 Useful options:
- `patience`: Number of epochs to wait before triggering
- `restore_best_weights=True`: Reverts to the best model

---

### 🧠 Transfer Learning

```python
from tensorflow.keras.applications import EfficientNetB4, VGG16, ResNet50, InceptionV3
```

Pretrained models on ImageNet:
- `EfficientNetB4`: Scalable, accurate, efficient
- `VGG16`: Classic architecture, easy to fine-tune
- `ResNet50`: Uses residual connections to go deeper
- `InceptionV3`: Parallel convolutions for multi-scale processing

These models are often used with `include_top=False` for custom classification heads.

---

### 🔄 TFRecord for Large Datasets

```python
from tensorflow.data import TFRecordDataset
```

- `TFRecordDataset`: Loads binary-encoded datasets efficiently
  - Faster than reading images individually
  - Recommended for large-scale training (e.g., cloud environments)

---

### 🔣 Probabilistic Modeling

```python
import tensorflow_probability as tfp
```

- `tensorflow_probability`: Useful for:
  - Custom probabilistic layers
  - Bayesian deep learning
  - Custom loss functions (e.g., KL divergence)

---

### 🧪 Model Evaluation (Post-training)

```python
from sklearn.metrics import f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
```

Used for interpreting performance beyond simple accuracy:

- `f1_score`: Harmonic mean of precision and recall
- `confusion_matrix`: Breakdown of prediction results
- `classification_report`: Precision, recall, f1-score per class
- `ConfusionMatrixDisplay`: Plot confusion matrix as heatmap

---
