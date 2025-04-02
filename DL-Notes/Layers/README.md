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


# Convolutional Neural Networks (CNNs) - Pooling & Activation Layers

## Pooling Layer
A Pooling Layer is an essential component of CNNs used to reduce the spatial dimensions (height and width) of feature maps while retaining important features. It enhances computational efficiency and reduces the risk of overfitting.

### Benefits of Pooling:
- **Reduces Dimensionality** → Decreases feature map size, leading to faster computations.
- **Extracts Dominant Features** → Keeps the most significant information while removing less important details.
- **Improves Model Generalization** → Prevents overfitting by reducing complexity.
- **Makes CNNs More Translation Invariant** → Recognizes features regardless of their position in the image.

### Types of Pooling Layers

#### (A) Max Pooling (Most Common)
- **Functionality:** Selects the maximum value from each window (region) of the feature map.
- **Purpose:** Retains the most important feature (strongest activation) while reducing size.

##### Example with a 2×2 filter and stride = 2:
###### Input Feature Map (4×4)
```
1   3   2   1  
4   6   5   7  
8   9   4   2  
3   5   1   0  
```
###### Max Pooling (2×2 filter)
```
6   7  
9   5  
```
- **Pros:** Preserves strong features and reduces computation.
- **Cons:** Ignores other activations, which may lose some information.

#### (B) Average Pooling
- **Functionality:** Takes the average value from each region of the feature map.

##### Example with a 2×2 filter and stride = 2:
###### Input Feature Map (4×4)
```
1   3   2   1  
4   6   5   7  
8   9   4   2  
3   5   1   0  
```
###### Average Pooling (2×2 filter)
```
3.5   3.75  
6.25  2.75  
```
- **Pros:** Retains more information about feature distribution.
- **Cons:** Less effective at filtering noise than Max Pooling.

#### (C) Global Pooling (Global Max/Average Pooling)
- **Functionality:** Reduces the entire feature map to a single value.
- **Usage:** Applied before fully connected layers to reduce parameters.
- **Example:** If the input feature map is 8×8, Global Max Pooling will take the maximum value from the entire 8×8 map and return a single value.
- **Commonly used in:** CNN architectures like ResNet and MobileNet.

### Key Pooling Parameters
| Parameter  | Description  |
|------------|-------------|
| **Pool Size** | Defines the window size (e.g., 2×2 or 3×3) |
| **Stride** | Defines how far the window moves (e.g., stride = 2 means it moves 2 pixels at a time) |
| **Padding** | "Same" (keeps size same) or "Valid" (shrinks size) |

---

## Activation Layer
### Purpose:
An activation layer applies a non-linear transformation to the input, helping the neural network learn complex patterns. Without activation functions, a neural network behaves like a simple linear regression model.

### Types of Activation Functions

#### **ReLU (Rectified Linear Unit)**
- **Most commonly used in CNNs**.
- **Function:** Replaces negative values with zero.
- **Formula:** \[ f(x) = \max(0, x) \]
- **Benefit:** Helps solve the vanishing gradient problem.
- **Usage:**
```python
layers.Activation('relu')
```

#### **Sigmoid**
- **Used for binary classification tasks**.
- **Function:** Maps values between 0 and 1.
- **Formula:** \[ f(x) = \frac{1}{1 + e^{-x}} \]
- **Usage:**
```python
layers.Activation('sigmoid')
```

#### **Tanh (Hyperbolic Tangent)**
- **Similar to Sigmoid but maps values between -1 and 1**.
- **Function:** Used in some cases to center data around zero.
- **Formula:** \[ f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \]
- **Usage:**
```python
layers.Activation('tanh')
```

#### **Softmax**
- **Used for multi-class classification problems**.
- **Function:** Converts logits into probabilities that sum to 1.
- **Usage:**
```python
layers.Activation('softmax')
```

---


## 2. Batch Normalization Layer

### Purpose:
Batch Normalization (BatchNorm) normalizes activations across mini-batches to speed up training and improve stability.

### Benefits:
✔ Normalizes input to each layer (mean ≈ 0, variance ≈ 1).
✔ Helps reduce internal covariate shift.
✔ Allows for higher learning rates, reducing training time.
✔ Acts as a regularizer, reducing the need for Dropout.

### How Batch Normalization Works:
1. Computes the mean and variance of activations in a batch.
2. Normalizes the activations:
   \[ x_{normalized} = \frac{x - \mu}{\sigma} \]
3. Applies a learnable scale (γ) and shift (β):
   \[ y = \gamma x_{normalized} + \beta \]

### BatchNorm in Keras:
```python
from tensorflow.keras import layers
layers.BatchNormalization()
```

---

## 3. Dropout Layer

### Purpose:
Dropout randomly drops neurons during training to reduce overfitting.

### How It Works:
- During training, randomly sets a fraction of activations to zero.
- This forces the network to not rely on specific neurons, making it more robust.
- During inference (testing), all neurons are used.

### Dropout in Keras:
```python
layers.Dropout(rate=0.5)  # 50% neurons randomly dropped
```

### Benefits:
✔ Reduces overfitting.
✔ Helps the network generalize better to unseen data.
✔ Often used in fully connected (Dense) layers.

---

## 4. Flatten Layer

### Purpose:
The Flatten layer converts a multi-dimensional feature map into a 1D vector, preparing it for the fully connected (Dense) layer.

### Example:
- Input feature map: \(7 \times 7 \times 64\) (7x7 image with 64 channels)
- Flattened to \(1 \times 3136\) \((7 \times 7 \times 64 = 3136)\)

### Flatten in Keras:
```python
layers.Flatten()
```

---

## 5. Fully Connected (FC) Layer / Dense Layer

### Purpose:
A Fully Connected (FC) Layer, also called a Dense Layer in Keras, is a layer where every neuron is connected to every neuron in the next layer. It is commonly used in the final stages of a Convolutional Neural Network (CNN) or as a key component in traditional Multi-Layer Perceptrons (MLPs).

### How It Works:
Each neuron in a Dense layer receives input from all neurons in the previous layer and produces an output for all neurons in the next layer.
It is mathematically equivalent to a matrix multiplication followed by an activation function:
\[ y = W \cdot x + b \]

Where:
- \( x \) = Input vector (flattened features from previous layers)
- \( W \) = Weight matrix (learnable parameters)
- \( b \) = Bias vector
- \( y \) = Output vector (activations)

Each neuron performs:
\[ z = \sum (w_i \cdot x_i) + b \]
Then, an activation function (like ReLU, Sigmoid, or Softmax) is applied to introduce non-linearity.

### Use Cases:
✅ In Convolutional Neural Networks (CNNs) → Converts extracted features into final predictions.
✅ In Deep Neural Networks (DNNs) → Used in classification, regression, and reinforcement learning tasks.
✅ In Recurrent Neural Networks (RNNs) → Used after LSTM/GRU layers for sequence tasks.

### Fully Connected Layer in Keras:
```python
from tensorflow.keras import layers

# Fully Connected Layer with 128 neurons and ReLU activation
layers.Dense(128, activation='relu')
```
