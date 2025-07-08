## ✅ **1. Model Architecture: LeNet-Inspired CNN**

### `Sequential([...])`

* **TensorFlow Sequential model** used to stack layers one after another.
* Easier to manage when the model is linear (one layer after another).

---

### **Preprocessing Layers:**

```python
Resizing(256,256),
Rescaling(1./255),
```
-
* **`Resizing(256, 256)`**: Resizes all input images to 256×256 pixels regardless of original size.
* **`Rescaling(1./255)`**: Normalizes pixel values from `[0, 255]` to `[0, 1]`.

---

### **Convolutional Block 1:**

```python
Conv2D(...),
BatchNormalization(),
MaxPool2D(...),
Dropout(...),
```

* **`Conv2D`**:

  * Applies a convolution operation.
  * Uses filters from `Configuration["NoOfFilters"]`, kernel size from `Configuration["KernelSize"]`, and stride from `Configuration["NoOfStrides"]`.
  * Padding is 'valid' (no zero-padding).
  * Activation is ReLU (non-linearity).
  * L2 regularization used to reduce overfitting (`Configuration["RegularizationRate"]`).

* **`BatchNormalization()`**: Normalizes activations in the batch to stabilize and speed up training.

* **`MaxPool2D`**:

  * Pool size from `Configuration["PoolSize"]`.
  * Stride is **2×NoOfStrides**, effectively reducing feature map dimensions.

* **`Dropout()`**: Randomly drops a fraction of nodes to prevent overfitting. Rate from `Configuration["DropOutRate"]`.

---

### **Convolutional Block 2:**

```python
Conv2D(...),
BatchNormalization(),
MaxPool2D(...),
```

* Similar to Block 1, but:

  * Number of filters is increased: `Configuration["NoOfFilters"] * 2 + 4`.
  * Extracts deeper features from the image.

---

### **Flatten + Fully Connected (Dense) Layers:**

```python
Flatten(),

Dense(...),
BatchNormalization(),
Dropout(...),

Dense(...),
BatchNormalization(),
```

* **`Flatten()`**: Converts 2D feature maps to 1D vector for dense layer input.
* **Dense Layers**:

  * Two fully connected layers:

    * First dense layer with `Configuration["NoOfUDense1"]` units.
    * Second with `Configuration["NoOfUDense2"]`.
  * Both use ReLU and L2 regularization.
* **Dropout**: Applied after the first dense layer.
* **BatchNormalization**: Applied after each dense layer to stabilize learning.

---

### **Output Layer:**

```python
Dense(Configuration["NoOfClasses"], activation="softmax"),
```

* Fully connected layer with number of neurons equal to number of emotion classes.
* Uses **softmax** for multi-class classification (outputs class probabilities).

---

## ✅ **2. Model Compilation**

```python
lenet_model.compile(
    optimizer=Adam(learning_rate=Configuration["LearningRate"]),
    loss=CategoricalCrossentropy(),
    metrics=[CategoricalAccuracy(), TopKCategoricalAccuracy(k=3)]
)
```

* **Optimizer**: Adam with custom learning rate.
* **Loss Function**: `CategoricalCrossentropy()` for multi-class classification.
* **Metrics**:

  * `CategoricalAccuracy`: How often predictions match labels.
  * `TopKCategoricalAccuracy(k=3)`: Accuracy considering top 3 predicted classes.

---

## ✅ **3. Training Aids: Callbacks**

### 📌 **EarlyStopping**

```python
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

* Monitors validation loss.
* Stops training if no improvement for 5 consecutive epochs.
* Restores weights from epoch with best val\_loss.

---

### 📌 **ModelCheckpoint**

```python
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
```

* Saves model to `.keras` file **only** when validation loss improves.
* Saves **entire model** (architecture + weights).
* Useful for resuming training or deployment.

---

## ✅ **4. Model Training or Loading**

### 👇 **Train model if not already trained:**

```python
if not train_again and os.path.exists('EmotionRecognitionSystemFinal1.h5'):
    model = tf.keras.models.load_model('EmotionRecognitionSystemFinal1.h5')
    print("Model loaded from file.")
else:
    ...
```

* If `train_again` is **False** and saved model file exists:

  * Model is loaded from file (`EmotionRecognitionSystemFinal1.h5`).
* Else:

  * Model is trained on `TrainingDS` with validation on `ValidationDS`.

---

### 🏋️‍♂️ **Training the model**

```python
history = lenet_model.fit(
    TrainingDS,
    epochs=Configuration["NoOfEpochs"],
    validation_data=ValidationDS,
    batch_size=Configuration["BatchSize"],
    steps_per_epoch=steps_per_epoch,
    callbacks=[early_stopping, checkpoint]
)
```

* `TrainingDS`: Training dataset.
* `ValidationDS`: Dataset for validation.
* Number of epochs from config.
* Batch size also from config.
* **`steps_per_epoch`**: Number of batches in each epoch = `23229 // BatchSize`.
* Uses both `early_stopping` and `checkpoint`.

---

### 💾 **Saving the model after training**

```python
lenet_model.save('EmotionRecognitionSystemFinal1.h5')
```

* Final trained model is saved for future use without retraining.

---

## ✅ **5. Summary of Configuration Keys**

These are dynamically accessed using `Configuration[...]`. Ensure `Configuration` dictionary includes:

| Key                    | Description                                   |
| ---------------------- | --------------------------------------------- |
| `"NoOfFilters"`        | Number of filters in the first Conv2D layer   |
| `"KernelSize"`         | Tuple defining convolution kernel size        |
| `"NoOfStrides"`        | Stride size in Conv2D and MaxPool             |
| `"RegularizationRate"` | L2 regularization factor                      |
| `"PoolSize"`           | Size of pooling window                        |
| `"DropOutRate"`        | Fraction of units to drop during training     |
| `"NoOfUDense1"`        | Number of units in first dense layer          |
| `"NoOfUDense2"`        | Number of units in second dense layer         |
| `"NoOfClasses"`        | Number of output classes (emotion categories) |
| `"LearningRate"`       | Learning rate for Adam optimizer              |
| `"NoOfEpochs"`         | Max number of training epochs                 |
| `"BatchSize"`          | Number of samples per training batch          |

---
