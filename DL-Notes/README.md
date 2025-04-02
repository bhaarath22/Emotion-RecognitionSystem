# ***Deep Learning***

## ***Table of Contents***
1. [Basics](#Basics)
2. ![Metrics](#Metrics-in-TensorFlow)
3. [Layers](#DL-Notes/Layers)
4. [Regularization](#DL-Notes/Regularization-Optimizers-Callbacks)
5. [Optimizers](#DL-Notes/Regularization-Optimizers-Callbacks)
6. [Callbacks](#DL-Notes/Regularization-Optimizers-Callbacks)
7. [TransferLearning](#DL-Notes/Transfer-Learning)
8. [Refferences](#References)


## ***Basics***

## *OpenCV*

OpenCV (Open Source Computer Vision Library) is a free, open-source library for computer vision and machine learning. It offers a wide range of algorithms and tools for image and video processing, object detection, and more. 

OpenCV was built to provide a common infrastructure for computer vision applications. The library has more than 2500 optimized algorithms, which include a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms.

**These algorithms can be used for**:

- Face detection and recognition
- Object identification
- Classifying human actions in videos
- Tracking camera movements and moving objects
- Extracting 3D models of objects
- Producing 3D point clouds from stereo cameras
- Stitching images together to create high-resolution panoramic images
- Finding similar images from an image database
- Removing red eyes from flash photography
- Eye movement tracking
- Scene recognition and augmented reality marker placement

```sh
pip install opencv-python
```


## ***NumPy***
NumPy is the fundamental package for scientific computing in Python. It provides a multidimensional array object and various derived objects (such as masked arrays and matrices) for numerical data manipulation.  

```sh
pip install numpy
```


## ***Pandas***
Pandas is a Python library used for working with datasets. It includes functions for analyzing, cleaning, exploring, and manipulating data, especially in CSV and spreadsheet formats.  

```sh
pip install pandas
```


## ***Matplotlib***
Matplotlib is a popular Python library used for data visualization. It enables the creation of static, animated, and interactive plots,  including:  

- Line graphs
- Bar charts
- Histograms
- Scatter plots

```sh
pip install matplotlib
```


## ***Seaborn***
Seaborn is a Python data visualization library built on top of Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics. It works well with Pandas DataFrames and simplifies complex visualizations.  

- **Heatmaps**: A type of data visualization where values are represented as colors on a grid.

  
```sh
pip install seaborn
```

---

## ***TensorFlow***  

TensorFlow is an open-source machine learning framework developed by Google. It is widely used for building deep learning models, neural networks, and large-scale machine learning applications.   

TensorFlow provides tools for training and deploying AI models across various platforms, including CPUs, GPUs, and TPUs.  

```sh
pip install tensorflow
```

**For GPU support (NVIDIA CUDA)**:
```sh
pip install tensorflow-gpu
```


## **Keras 3**
Keras is a high-level deep learning API that works interchangeably with TensorFlow, JAX, and PyTorch. It simplifies the process of building, training, and deploying neural networks. Keras is user-friendly and modular.


```sh
pip install keras
```

### ***Sequential and Functional APIs***:  

Keras offers two main ways to define models:  

***Sequential API***: Ideal for building simple, linear models where layers are stacked sequentially.   

***Functional API***: More flexible, allowing for complex model architectures with non-linear connections and shared layers. 

## ***Node (Neuron)***  

- A node (neuron) is a fundamental unit in a neural network that processes input and passes it forward.  

- Each node in a Dense layer receives input from all nodes of the previous layer.  

- Each node applies a weight, bias, and an activation function to compute an output.

### **Mathematical Representation**:  

For each neuron:
\[ y = \text{activation}(W \cdot X + b) \]

Where:
- \( X \) = Input from the previous layer
- \( W \) = Weights (learnable parameters)
- \( b \) = Bias
- \( \text{activation} \) = Activation function (ReLU, Sigmoid, Softmax)
- \( y \) = Output value
----  

## ***Metrics in TensorFlow***  

Metrics help evaluate a model’s performance during training and validation.   

Metric functions are similar to loss functions, except that the results from evaluating a metric are not used when training the model.   

A metric is a function that is used to judge the performance of a model.  


---
##  **Classification Metrics**  

### *Accuracy*  

- Measures the fraction of correctly classified images.
- Works well for balanced datasets.
  - Use: `metrics=['accuracy']`
---
## **Classification Metrics**

### **Precision, Recall, and F1-Score**

#### **Precision**
Measures how many of the predicted positives are actually correct:

#### **Recall (Sensitivity)**
Measures how many actual positives were correctly predicted:


#### **F1-Score**
Harmonic mean of precision and recall, useful when data is imbalanced:

```python
from tensorflow.keras.metrics import Precision, Recall
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])
```

---
## **AUC (Area Under the ROC Curve)**

Measures the ability of the model to distinguish between classes:

```python
from tensorflow.keras.metrics import AUC
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC()])
```

---
## **Regression Metrics**

### **Mean Squared Error (MSE)**
Penalizes larger errors more than small ones:

```python
from tensorflow.keras.metrics import MeanSquaredError
model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])
```

### **Mean Absolute Error (MAE)**
Measures absolute differences, less sensitive to outliers than MSE:

```python
from tensorflow.keras.metrics import MeanAbsoluteError
model.compile(optimizer='adam', loss='mae', metrics=[MeanAbsoluteError()])
```

---
## **R² Score (Coefficient of Determination)**
Measures how well the model explains the variance in the data:

  
```python
import tensorflow.keras.backend as K

def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

model.compile(optimizer='adam', loss='mse', metrics=[r2_score])
```

---


## ***References***
1. [OpenCV](https://opencv.org/about/)
2. [Pandas](https://www.w3schools.com/python/pandas/pandas_intro.asp)
3. [Matplotlib](https://matplotlib.org)
4. [Seaborn](https://seaborn.pydata.org)
5. [TensorFlow](https://www.tensorflow.org/learn)
6. [Keras](https://keras.io/getting_started/intro_to_keras_for_engineers/)
7. [metrices](https://keras.io/api/metrics/)

