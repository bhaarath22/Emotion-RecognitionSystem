# Deep Learning

## OpenCV
OpenCV (Open Source Computer Vision Library) is a free, open-source library for computer vision and machine learning. It offers a wide range of algorithms and tools for image and video processing, object detection, and more. OpenCV was built to provide a common infrastructure for computer vision applications. The library has more than 2500 optimized algorithms, which include a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms.

These algorithms can be used for:
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

## Installation
To install the required libraries, use the following commands:

### OpenCV
```sh
pip install opencv-python
```

### NumPy
NumPy is the fundamental package for scientific computing in Python. It provides a multidimensional array object and various derived objects (such as masked arrays and matrices) for numerical data manipulation.
```sh
pip install numpy
```

### Pandas
Pandas is a Python library used for working with datasets. It includes functions for analyzing, cleaning, exploring, and manipulating data, especially in CSV and spreadsheet formats.
```sh
pip install pandas
```

### Matplotlib
Matplotlib is a popular Python library used for data visualization. It enables the creation of static, animated, and interactive plots, including:
- Line graphs
- Bar charts
- Histograms
- Scatter plots

```sh
pip install matplotlib
```

### Seaborn
Seaborn is a Python data visualization library built on top of Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics. It works well with Pandas DataFrames and simplifies complex visualizations.

#### Example Visualization:
- **Heatmaps**: A type of data visualization where values are represented as colors on a grid.
```sh
pip install seaborn
```

## TensorFlow
TensorFlow is an open-source machine learning framework developed by Google. It is widely used for building deep learning models, neural networks, and large-scale machine learning applications. TensorFlow provides tools for training and deploying AI models across various platforms, including CPUs, GPUs, and TPUs.
```sh
pip install tensorflow
```
For GPU support (NVIDIA CUDA):
```sh
pip install tensorflow-gpu
```

### Keras 3
Keras is a high-level deep learning API that works interchangeably with TensorFlow, JAX, and PyTorch. It simplifies the process of building, training, and deploying neural networks. Keras is user-friendly and modular.
```sh
pip install keras
```

## Deep Learning Layers
### 1. Convolutional Layer
A Convolutional Layer (Conv Layer) is the core building block of a Convolutional Neural Network (CNN). It is responsible for detecting features such as edges, textures, shapes, and patterns in an image. The Convolutional Layer applies a small filter (kernel) to an input image or feature map to extract important features.

#### Steps in a Convolutional Layer:
1. **Kernel (Filter) Application** – A small matrix (e.g., 3×3 or 5×5) slides over the input image.
2. **Element-wise Multiplication** – Each filter value is multiplied with corresponding pixel values.
3. **Summation** – The results are summed to get a single value for that region.
4. **Feature Map Generation** – The process repeats across the image, creating a new feature map.

#### Important Parameters in a Convolutional Layer:
- **Kernel Size** (e.g., 3×3 or 5×5) – Defines the size of the filter.
- **Stride** – Controls how much the filter moves:
  - A stride of 1 moves pixel by pixel.
  - A stride of 2 skips every other pixel.
- **Padding** – Helps maintain the input size by adding zeros around the edges:
  - **Valid Padding**: No padding (output size shrinks).
  - **Same Padding**: Adds zeros so the output size matches the input size.

## References
1. [OpenCV](https://opencv.org/about/)
2. [Pandas](https://www.w3schools.com/python/pandas/pandas_intro.asp)
3. [Matplotlib](https://matplotlib.org)
4. [Seaborn](https://seaborn.pydata.org)
5. [TensorFlow](https://www.tensorflow.org/learn)
6. [Keras](https://keras.io/getting_started/intro_to_keras_for_engineers/)

