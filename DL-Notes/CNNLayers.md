

### **1. Convolutional Layer (`Conv2D`)**
**Role**: Extracts spatial features (edges, textures, patterns) using learnable kernels.  
**Key Parameters**:
- `filters` (int): Number of output channels (e.g., 32, 64).
- `kernel_size` (int/tuple): Size of the convolution window (e.g., `(3, 3)`).
- `strides` (int/tuple): Step size of the kernel (default `(1, 1)`).
- `padding` (str): `'valid'` (no padding) or `'same'` (output size = input size).
- `activation` (str): Nonlinearity function (e.g., `'relu'`).
- `input_shape` (tuple): Required for the first layer (e.g., `(224, 224, 3)` for RGB images).

**Example**:
```python
from tensorflow.keras.layers import Conv2D

Conv2D(
    filters=64,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    activation='relu',
    input_shape=(224, 224, 3)
)
```

---

### **2. Pooling Layers**
#### **MaxPooling2D**
**Role**: Downsamples feature maps by taking the maximum value in each window, reducing spatial dimensions.  
**Parameters**:
- `pool_size` (int/tuple): Window size (e.g., `(2, 2)`).
- `strides` (int/tuple): Step size (defaults to `pool_size`).
- `padding` (str): `'valid'` or `'same'`.

**Example**:
```python
from tensorflow.keras.layers import MaxPooling2D

MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')
```

#### **AveragePooling2D**
**Role**: Downsamples by averaging values in each window (less common than max pooling).  
**Parameters**: Same as `MaxPooling2D`.

---

### **3. Batch Normalization (`BatchNormalization`)**
**Role**: Stabilizes training by normalizing activations (zero mean, unit variance).  
**Parameters**:
- `axis` (int): Axis to normalize (default `-1` for channels-last).
- `momentum` (float): Momentum for moving statistics (default `0.99`).
- `epsilon` (float): Small value to avoid division by zero (default `0.001`).

**Example**:
```python
from tensorflow.keras.layers import BatchNormalization

BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)
```

---

### **4. Dropout (`Dropout`)**
**Role**: Prevents overfitting by randomly setting a fraction of inputs to zero.  
**Parameters**:
- `rate` (float): Fraction to drop (e.g., `0.5` for 50%).

**Example**:
```python
from tensorflow.keras.layers import Dropout

Dropout(rate=0.5)
```

---

### **5. Flatten (`Flatten`)**
**Role**: Converts multi-dimensional features to 1D for dense layers.  
**No parameters**.

**Example**:
```python
from tensorflow.keras.layers import Flatten

Flatten()
```

---

### **6. Dense (`Dense`)**
**Role**: Fully connected layer for classification/regression.  
**Parameters**:
- `units` (int): Number of neurons.
- `activation` (str): e.g., `'relu'`, `'softmax'`.

**Example**:
```python
from tensorflow.keras.layers import Dense

Dense(units=1024, activation='relu')
```

---

### **7. Global Pooling Layers**
#### **GlobalAveragePooling2D**
**Role**: Reduces each feature map to a single value by averaging (replaces `Flatten`).  
**No parameters**.

**Example**:
```python
from tensorflow.keras.layers import GlobalAveragePooling2D

GlobalAveragePooling2D()
```

#### **GlobalMaxPooling2D**
**Role**: Similar but takes the maximum value.  

---

### **8. Activation Layers**
**Role**: Applies nonlinearity (can be separate or part of `Conv2D`/`Dense`).  
**Parameters**:
- `activation` (str): e.g., `'relu'`, `'sigmoid'`, `'softmax'`.

**Example**:
```python
from tensorflow.keras.layers import Activation

Activation('relu')
```

---

### **9. ZeroPadding2D**
**Role**: Adds zeros around the image (e.g., to handle border effects).  
**Parameters**:
- `padding` (int/tuple): e.g., `(1, 1)` adds 1 pixel on all sides.

**Example**:
```python
from tensorflow.keras.layers import ZeroPadding2D

ZeroPadding2D(padding=(1, 1))
```

---

### **10. DepthwiseConv2D**
**Role**: Applies a separate convolution to each input channel (used in lightweight models like MobileNet).  
**Parameters**: Similar to `Conv2D` but no `filters` (output depth = input depth × `depth_multiplier`).

**Example**:
```python
from tensorflow.keras.layers import DepthwiseConv2D

DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')
```

---

### **11. UpSampling2D**
**Role**: Increases spatial dimensions (e.g., for segmentation/autoencoders).  
**Parameters**:
- `size` (int/tuple): Scaling factor (e.g., `(2, 2)`).

**Example**:
```python
from tensorflow.keras.layers import UpSampling2D

UpSampling2D(size=(2, 2))
```

---

### **12. Concatenate (`Concatenate`)**
**Role**: Merges multiple inputs (e.g., in U-Net, Inception).  
**Parameters**:
- `axis` (int): Axis to concatenate (default `-1`).

**Example**:
```python
from tensorflow.keras.layers import Concatenate

Concatenate(axis=-1)([input1, input2])
```

---

### **Example: Full CNN Model (Functional API)**
```python
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(224, 224, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs, outputs)
```

---

### **Key Takeaways**:
- **Convolutional/Pooling**: Feature extraction + downsampling.
- **BatchNorm/Dropout**: Stabilize training.
- **Global Pooling**: Alternative to `Flatten`.
- **Functional API**: Required for complex architectures (skip connections, multi-input/output).
