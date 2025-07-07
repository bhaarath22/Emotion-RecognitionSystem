## ✅ **Step 1: Evaluate the Model**

```python
results = lenet_model.evaluate(TestingDS, verbose=0)
loss = results[0]
accuracy = results[1]
top_k_accuracy = results[2]
```

* **`lenet_model.evaluate()`**:

  * Evaluates the model on the **TestingDS** dataset.
  * Returns loss and metrics defined during compilation:

    * `CategoricalAccuracy` → `results[1]`
    * `TopKCategoricalAccuracy(k=3)` → `results[2]`

```python
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")
print(f"Test top-k accuracy: {top_k_accuracy}")
```

* **Prints results** to check how well the model performs on the test set.

---

## ✅ **Step 2: Plot Training History (Accuracy & Loss Curves)**

### **Fetch training history:**

```python
history_dict = history.history
```

* `history.history` is a dictionary containing metric values per epoch:

  * `'accuracy'`, `'val_accuracy'`, `'loss'`, `'val_loss'`

### **Plotting Accuracy:**

```python
plt.subplot(1, 2, 1)
plt.plot(history_dict['accuracy'], label='Training Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
```

* Visualizes how accuracy evolved during training vs validation.
* Helps spot **underfitting** or **overfitting**.

### **Plotting Loss:**

```python
plt.subplot(1, 2, 2)
plt.plot(history_dict['loss'], label='Training Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
```

* Visualizes how loss changed over epochs.
* Large gap between training and validation loss may indicate overfitting.

---

## ✅ **Step 3: F1 Score and Confusion Matrix**

### **Predict on Testing Data**

```python
y_true = []
y_pred = []
for images, labels in TestingDS:
    predictions = lenet_model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))
```

* Loops through batches in `TestingDS`.
* Gets **true labels** and **model predictions**.
* Converts one-hot encoded labels and softmax outputs to class indices using `argmax()`.

---

### **Calculate F1 Score**

```python
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
```

* **`f1_score`**: Harmonic mean of precision and recall.
* **`average='weighted'`**:

  * Takes class imbalance into account by weighting F1 by support (number of samples in each class).
  * Suitable for multi-class classification.

---

### **Classification Report**

```python
print(classification_report(y_true, y_pred))
```

* Outputs:

  * **Precision**: TP / (TP + FP)
  * **Recall**: TP / (TP + FN)
  * **F1-score**
  * **Support**: Number of true samples for each class.

---

### **Confusion Matrix**

```python
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ClassNames)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
```

* Shows how many predictions were made correctly and incorrectly for each class.
* Helps identify **confusable** emotion classes (e.g., Angry vs Disgust).

---

## ✅ **What to Ensure Before Running This Section**

1. ✅ `TestingDS` must be a valid `tf.data.Dataset` with image-label pairs.
2. ✅ `ClassNames` must be defined and match label indices.

   ```python
   ClassNames = ['Angry', 'Happy', 'Sad', 'Neutral', ...]  # as per your dataset
   ```
3. ✅ Required imports must be present:

   ```python
   from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
   import matplotlib.pyplot as plt
   import numpy as np
   ```

---

## ✅ **Optional Improvements**

* **Normalize Confusion Matrix** for better interpretability:

  ```python
  disp.plot(cmap='Blues', values_format='.2f', normalize='true')
  ```

* **Save Plots**:

  ```python
  plt.savefig("training_accuracy_loss.png")
  ```

* **Show class-wise F1 in a bar plot** for better comparison.

---

## ✅ **Final Takeaway**

This section completes the model lifecycle:

1. **Evaluates** performance quantitatively (accuracy, loss, top-k accuracy).
2. **Visualizes** training behavior (accuracy and loss curves).
3. **Analyzes** classification performance (F1 score, class-level metrics, confusion matrix).
