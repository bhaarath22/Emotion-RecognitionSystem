## ***Regularization Techniques***  

Regularization techniques help prevent overfitting by adding constraints to the model’s parameters, ensuring better generalization to unseen data. TensorFlow provides built-in regularizers that can be applied to convolutional and dense layers.

### 1. L1 and L2 Regularization (Weight Decay)
These regularizers work by adding a penalty term to the loss function.

#### (i) L1 Regularization (Lasso)
- Adds the absolute values of the weights (|w|) to the loss function.
- Encourages sparsity, meaning some weights become zero, reducing model complexity.
- Formula:
  \[ L1\_Loss = \lambda \sum |w| \]

#### (ii) L2 Regularization (Ridge)
- Adds the squared magnitude of weights (w²) to the loss function.
- Encourages small, non-zero weights, preventing overfitting.
- Formula:
  \[ L2\_Loss = \lambda \sum w^2 \]

#### (iii) L1 + L2 (Elastic Net)
- Combines both L1 and L2 regularization.
- Formula:
  \[ Loss = \lambda_1 \sum |w| + \lambda_2 \sum w^2 \]

### 2. Dropout Regularization
- Randomly drops units during training to reduce reliance on specific neurons, improving generalization.

### 3. Batch Normalization (Indirect Regularization)
- Normalizes activations to stabilize training and improve performance.

### 4. Data Augmentation (Indirect Regularization)
- Artificially increases the training data by applying transformations (e.g., rotation, flipping, cropping) to improve model robustness.

--- 

## ***Optimizers***
Optimizers play a crucial role in training CNNs by adjusting the network’s weights to minimize the loss function.

### 1. Gradient Descent (GD)
#### Types:
- **Batch GD**: Uses the entire dataset to compute gradients.
- **Stochastic GD (SGD)**: Uses one sample at a time for updates.
- **Mini-batch GD**: Uses small batches for balanced efficiency.

### 2. Optimizers with Momentum
- **Momentum Optimizer**: Accelerates convergence and reduces oscillations.

### 3. Adaptive Optimizers
#### (i) Adagrad (Adaptive Gradient Algorithm)
- Adjusts learning rates based on past gradients.
- Works well for sparse data.

#### (ii) RMSprop (Root Mean Square Propagation)
- Modifies Adagrad by using an exponentially decaying average.

#### (iii) Adam (Adaptive Moment Estimation)
- Combines Momentum and RMSprop. 
#### (iv) AdamW (Adam with Weight Decay)
- Reduces overfitting by incorporating weight decay.

### 4. Other Advanced Optimizers
#### (i) Nadam (Nesterov-accelerated Adaptive Moment Estimation)
- Combines Adam with Nesterov Momentum for faster convergence.

#### (ii) LAMB (Layer-wise Adaptive Moments)
- Suitable for large-batch training.

### ***Optimizer Comparison***
| Optimizer | Learning Rate Adaptation | Convergence Speed | Suitable for |
|-----------|-------------------------|-------------------|--------------|
| SGD       | No                      | Slow              | General cases |
| Momentum  | No                      | Faster than SGD   | Deep CNNs, ResNets |
| RMSprop   | Yes                     | Medium-fast       | RNNs, deep networks |
| Adam      | Yes                     | Fast              | Most CNNs, image classification |
| AdamW     | Yes                     | Fast              | CNNs with regularization |
| Nadam     | Yes                     | Very fast         | Advanced CNN applications |
-------------------------------------------------------------------------------------------------------------

---

##  ***Callbacks***  

Callbacks are functions that help monitor and control the training process by saving models, stopping early, adjusting learning rates dynamically, and improving overall model efficiency.

### **Key Callback Function**

#### 1. **Early Stopping**
- Stops training when the validation loss stops improving.
- Prevents overfitting and reduces unnecessary training time.

**Example (Keras):**
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```
- `monitor`: Metric to track (e.g., 'val_loss', 'accuracy').
- `patience`: Number of epochs to wait before stopping.
- `restore_best_weights`: Loads the best weights before stopping.

#### 2. **Model Checkpoint**
- Saves the model during training based on a chosen metric, ensuring that the best model is retained.

**Example (Keras):**
```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
```
- `filepath`: File path to save the model.
- `monitor`: Metric to track.
- `save_best_only`: Saves only when the model improves.

#### 3. **Learning Rate Scheduler**
- Dynamically adjusts the learning rate during training to improve convergence.

**Example (Keras):**
```python
from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    return 0.001 * (0.1 ** (epoch // 10))

lr_scheduler = LearningRateScheduler(lr_schedule)
```
**Strategies:**
- **Step Decay:** Reduce the learning rate at fixed intervals.
- **Exponential Decay:** Gradually decrease the learning rate.
- **ReduceLROnPlateau:** Reduce learning rate when validation performance plateaus.

#### 4. **ReduceLROnPlateau**
- Reduces the learning rate when a monitored metric stops improving.

**Example (Keras):**
```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
```
- `factor`: Reduction factor for the learning rate.
- `patience`: Number of epochs to wait before reducing the learning rate.
- `min_lr`: Minimum allowable learning rate.

#### 5. **TensorBoard Callback**
- Provides visualizations for loss, accuracy, and other metrics.

**Example (Keras):**
```python
from tensorflow.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir='logs', histogram_freq=1)
```
- `log_dir`: Directory where logs are stored.
- `histogram_freq`: Frequency of logging weight histograms.

#### 6. **Custom Callbacks**
- Allows implementing custom logic during training.

**Example (Keras):**
```python
from tensorflow.keras.callbacks import Callback

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: Loss = {logs['loss']}")

custom_callback = CustomCallback()
```
