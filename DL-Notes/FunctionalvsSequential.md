
# ✅ Notes: **Keras Functional API vs Sequential API**

---

## 🚩 **1. Sequential API** (Simple and Straightforward)

### ➡️ What is it?
- The **easiest** way to build models in Keras.
- **Layers are stacked one after another**, like building a tower (no branching, no multiple inputs).

### ➡️ Syntax:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

### ➡️ Best For:
- **Simple models** (MLPs, basic CNNs).
- Straight **layer stack** models.

### ➡️ Limitations:
- Cannot handle **multiple inputs/outputs**.
- Cannot build models with **skip connections** or **branches** (like ResNet, Inception).

---

## 🚩 **2. Functional API** (Flexible and Powerful)

### ➡️ What is it?
- More **advanced** way of defining models.
- Uses a **graph structure** — data ("tensors") flow from layer to layer.
- Allows **complex models** with branches, multiple inputs, and outputs.

### ➡️ Syntax:
```python
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

input = Input(shape=(256, 256, 3))
x = Conv2D(32, (3,3), activation='relu')(input)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=input, outputs=output)
```

### ➡️ Best For:
- **Complex models**:
  - VGG
  - ResNet (skip connections)
  - Inception (branches)
  - GANs (multiple outputs)
- Models with:
  - **Multiple inputs**
  - **Multiple outputs**

---

## 🔥 **Differences Table**

| Feature               | **Sequential API**           | **Functional API**          |
|-----------------------|-------------------------------|-----------------------------|
| Layers are stacked    | Yes, one after another 🧱     | Yes, but can also **branch** 🔀 |
| Multiple Inputs       | ❌ Not possible               | ✅ Possible                  |
| Multiple Outputs      | ❌ Not possible               | ✅ Possible                  |
| Skip Connections      | ❌ Not possible               | ✅ Possible                  |
| Easy to start         | ✅ Beginner-friendly          | 🟡 Requires little more setup|
| Used in Research      | ❌ Rare                       | ✅ Standard                  |

---

## 🧠 **Pro Tip:**
- Start with **Sequential** if you’re a beginner.
- Move to **Functional API** if you want to build real-world models (e.g., VGG16, ResNet, U-Net, Transformers).

---

## 🎯 **When to use what?**
| Use **Sequential** if:          | Use **Functional API** if:          |
|---------------------------------|-------------------------------------|
| Model is simple and linear      | Model needs branches/skip paths     |
| No multiple inputs/outputs      | Multiple inputs/outputs required    |
| You're learning basic CNNs/MLPs | You're implementing research models |

---

## ✅ **Conclusion**
- **Sequential API** is like stacking LEGO bricks in a straight line.
- **Functional API** is like drawing a **flowchart** with multiple paths and connections.

For serious model building (like VGG, ResNet, etc.), **Functional API** is the **industry standard**.  
It offers full flexibility while still using Keras’ clean syntax.

---
