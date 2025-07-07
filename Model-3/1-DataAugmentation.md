### ✅ Code to Load Dataset

```python
TrainingDS = tf.keras.utils.image_dataset_from_directory(
    directory='.../Dataset/SplittedDS/train',
    image_size=(ImageSize, ImageSize),
    class_names=ClassNames,
    batch_size=32,
    shuffle=True,
    seed=90,
    label_mode='categorical'
)

ValidationDS = tf.keras.utils.image_dataset_from_directory(
    directory='.../Dataset/SplittedDS/val',
    image_size=(ImageSize, ImageSize),
    class_names=ClassNames,
    batch_size=32,
    shuffle=True,
    seed=90,
    label_mode='categorical'
)

TestingDS = tf.keras.utils.image_dataset_from_directory(
    directory='.../Dataset/SplittedDS/test',
    image_size=(ImageSize, ImageSize),
    class_names=ClassNames,
    batch_size=32,
    shuffle=False,
    label_mode='categorical'
)
```
## 🔍 What is `tf.keras.utils.image_dataset_from_directory`?

This function is a **convenient API** provided by TensorFlow that automatically:

* Loads images from a folder-based structure
* Labels them based on their directory names
* Converts them into a batched `tf.data.Dataset` (perfect for training with `model.fit`)

It is especially useful when:

*  dataset is organized into folders by class
* when we a need quick and efficient pipeline without writing a custom loader

## 🧾 Parameter Explanation (based on your usage)

```python
TrainingDS = tf.keras.utils.image_dataset_from_directory(
    directory='.../Dataset/SplittedDS/train',  # 📁 Root folder where image folders per class are present
    image_size=(ImageSize, ImageSize),         # 📐 Resize all images to this size (e.g., 224x224)
    class_names=ClassNames,                    # 📋 Optional list of class names to control the label order
    batch_size=32,                             # 📦 Number of images per batch
    shuffle=True,                              # 🔀 Shuffle the dataset after loading
    seed=90,                                   # 🌱 Fixes the shuffle order for reproducibility
    label_mode='categorical'                   # 🏷️ Format of labels — returns one-hot encoded vectors
)
```
### 🏷️ `label_mode` options:

* `'int'` → returns integer labels (e.g., 0, 1, 2)
* `'categorical'` → returns one-hot encoded labels (e.g., `[0, 1, 0]`)
* `'binary'` → for binary classification (0 or 1)
* `None` → no labels are returned
---

## ⚙️ Configuration Dictionary

All important hyperparameters are kept in one `Configuration` dictionary:

```python
Configuration = {
    "NoOfClasses": 8,
    "ClassNames": ClassNames,
    "ImageSize": ImageSize,
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
    "PatchSize": 16,  # (for transformer models)
    "ProjDim": 769     # (for transformer models)
}
```

---

## 🔁 Data Augmentation Pipeline

### 📦 1. Built-in Keras Layers (in `AugmentLayers`)

```python
AugmentLayers = Sequential([
    RandomRotation(factor=(-0.025, 0.025)),
    RandomFlip(mode='horizontal'),
    RandomContrast(factor=0.1),
    RandomBrightness(factor=(0.1, 0.2)),
    RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1))
])
```

### 💥 2. Custom Augmentations

#### ➕ Gaussian Noise

```python
def AddGaussianNoise(image, mean=0.0, stddev=1.0):
    noise = tf.random.normal(shape=tf.shape(image), mean=mean, stddev=stddev, dtype=image.dtype)
    return image + noise
```

#### 🎨 Color Jitter

```python
def ColorJitter(image, brightness=0.1, contrast=0.1):
    image = tf.image.random_brightness(image, max_delta=brightness)
    image = tf.image.random_contrast(image, lower=1-contrast, upper=1+contrast)
    return image
```

#### 📈 Final AugmentLayer Function

Applies built-in, noise, and color jitter together.

```python
def AugmentLayer(image, label):
    image = AugmentLayers(image, training=True)
    image = AddGaussianNoise(image)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = ColorJitter(image)
    return image, label
```

Use this in the training pipeline via:

```python
TrainingDS = TrainingDS.map(AugmentLayer)
ValidationDS = ValidationDS.prefetch(tf.data.AUTOTUNE)
```

---

## ✂️ CutMix Augmentation

**CutMix** is a strong augmentation technique where patches are cut and pasted among training images, and the ground truth labels are also mixed proportionally.

### 🔲 Bounding Box Generator

```python
def box(lambda_value, image_size):
    r_x = tf.cast(tfp.distributions.Uniform(0, image_size).sample(1)[0], tf.int32)
    r_y = tf.cast(tfp.distributions.Uniform(0, image_size).sample(1)[0], tf.int32)
    r_w = tf.cast(image_size * tf.math.sqrt(1 - lambda_value), tf.int32)
    r_h = tf.cast(image_size * tf.math.sqrt(1 - lambda_value), tf.int32)
    
    r_x = tf.clip_by_value(r_x - r_w // 2, 0, image_size - 1)
    r_y = tf.clip_by_value(r_y - r_h // 2, 0, image_size - 1)
    
    x_b_r = tf.clip_by_value(r_x + r_w, 0, image_size)
    y_b_r = tf.clip_by_value(r_y + r_h, 0, image_size)

    r_w = tf.maximum(x_b_r - r_x, 1)
    r_h = tf.maximum(y_b_r - r_y, 1)

    return r_y, r_x, r_h, r_w
```

### ✂️ CutMix Function

```python
def cutmix(TrainingDS1, TrainingDS2):
    (image_1, label_1), (image_2, label_2) = TrainingDS1, TrainingDS2
    lambda_value = tfp.distributions.Beta(0.2, 0.2).sample(1)[0]

    r_y, r_x, r_h, r_w = box(lambda_value, ImageSize)

    crop_2 = tf.image.crop_to_bounding_box(image_2, r_y, r_x, r_h, r_w)
    pad_2 = tf.image.pad_to_bounding_box(crop_2, r_y, r_x, ImageSize, ImageSize)

    crop_1 = tf.image.crop_to_bounding_box(image_1, r_y, r_x, r_h, r_w)
    pad_1 = tf.image.pad_to_bounding_box(crop_1, r_y, r_x, ImageSize, ImageSize)

    image = image_1 - pad_1 + pad_2

    lambda_value = tf.cast(1 - (r_w * r_h) / (ImageSize * ImageSize), tf.float32)
    label = lambda_value * tf.cast(label_1, tf.float32) + (1 - lambda_value) * tf.cast(label_2, tf.float32)

    return image, label
```

---

## 🛠️ Next Steps

* ✅ Model creation (e.g., custom CNN or transformer-based architecture)
* ⏳ Add early stopping, checkpoint saving
* 📊 Evaluate on TestingDS
* 📈 Plot training vs validation loss/accuracy
* 🧪 Test CutMix within the training loop
