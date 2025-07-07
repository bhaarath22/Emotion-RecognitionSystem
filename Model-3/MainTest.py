import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# TensorFlow and Keras Imports
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (Resizing, Rescaling, InputLayer, Input, Conv2D, BatchNormalization,
                                     MaxPool2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, Add,
                                     GlobalAveragePooling2D, Layer, RandomRotation, RandomFlip, RandomContrast,
                                     RandomBrightness, RandomTranslation)  # Augmentation layers
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Transfer Learning Architectures
from tensorflow.keras.applications import EfficientNetB4, VGG16, ResNet50, InceptionV3
# TFRecord Support
from tensorflow.core.example.feature_pb2 import Feature, Features
from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import BytesList, FloatList, Int64List
from tensorflow.io import TFRecordWriter, encode_jpeg, parse_single_example, decode_jpeg
from tensorflow.image import convert_image_dtype
from tensorflow.data import TFRecordDataset

# TensorFlow Probability
import tensorflow_probability as tfp

# TensorFlow Addons for Extra Metrics
#from tensorflow_addons.metrics import F1Score
from tensorflow.keras.metrics import F1Score
# Advanced Visualization
import plotly.express as px
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
# Lets Start
ClassNames =['anger','contempt','disgust','fear','happy','neutral','sad','surprise']
ImageSize=256;

# Initializing Training and Validation Datasets
#tf.random.set_seed(42)  # Set seed for reproducibility when visualizing the images to ignore displaying smae images every time
TrainingDS=tf.keras.utils.image_dataset_from_directory(
    directory='/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/ERSFinal/Dataset/SplittedDS/train',
    image_size=(ImageSize,ImageSize),
    class_names=ClassNames,
    batch_size=32,
    shuffle=True,
    seed=90,
    label_mode='categorical',


)
ValidationDS=tf.keras.utils.image_dataset_from_directory(
    directory='/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/ERSFinal/Dataset/SplittedDS/val',
    image_size=(ImageSize,ImageSize),
    class_names=ClassNames,
    batch_size=32,
    shuffle=True,
    seed=90,
    label_mode='categorical'
)
TestingDS=tf.keras.utils.image_dataset_from_directory(
    directory='/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/ERSFinal/Dataset/SplittedDS/test',
    image_size=(ImageSize,ImageSize),
    class_names=ClassNames,
    batch_size=32,
    shuffle=False,
    label_mode='categorical'
)
# Print dataset information

# Configuration dictionary for model parameters
Configuration = {
    "NoOfClasses": 8,              # Total number of classes
    "ClassNames": ClassNames,
    "ImageSize": ImageSize,
    "KernelSize": 3,                # Size of the convolutional kernel
    "BatchSize": 32,                # Number of samples per gradient update
    "LearningRate": 1e-3,           # Learning rate for the optimizer
    "NoOfEpochs": 25,               # Total number of training epochs
    "NoOfFilters": 6,               # Number of filters in the convolutional layers
    "NoOfUDense1": 1024,            # Number of units in the first dense layer
    "NoOfUDense2": 128,             # Number of units in the second dense layer
    "DropOutRate": 0.2,             # Dropout rate to prevent overfitting
    "RegularizationRate": 0.0,      # Regularization rate (L2)
    "NoOfStrides": 1,               # Step size for moving the filter over the input
    "PoolSize": 3,                  # Size of the pooling layers
    "PatchSize": 16,                # Used in vision transformers
    "ProjDim": 769                  # Projection dimension for transformer architectures
}

lenet_model = Sequential(
    [
       Resizing(256,256),
       Rescaling(1./255),

        Conv2D(filters=Configuration["NoOfFilters"], kernel_size=Configuration["KernelSize"],
               strides=Configuration["NoOfStrides"], padding='valid',
               activation='relu', kernel_regularizer=L2(Configuration["RegularizationRate"])),
        BatchNormalization(),
        MaxPool2D(pool_size=Configuration["PoolSize"], strides=Configuration["NoOfStrides"] * 2),
        Dropout(rate=Configuration["DropOutRate"]),

        Conv2D(filters=Configuration["NoOfFilters"] * 2 + 4, kernel_size=Configuration["KernelSize"],
               strides=Configuration["NoOfStrides"], padding='valid',
               activation='relu', kernel_regularizer=L2(Configuration["RegularizationRate"])),
        BatchNormalization(),
        MaxPool2D(pool_size=Configuration["PoolSize"], strides=Configuration["NoOfStrides"] * 2),

        Flatten(),

        Dense(Configuration["NoOfUDense1"], activation="relu",
              kernel_regularizer=L2(Configuration["RegularizationRate"])),
        BatchNormalization(),
        Dropout(rate=Configuration["DropOutRate"]),

        Dense(Configuration["NoOfUDense2"], activation="relu",
              kernel_regularizer=L2(Configuration["RegularizationRate"])),
        BatchNormalization(),

        Dense(Configuration["NoOfClasses"], activation="softmax"),
    ]
)

# lenet_model.summary()
# After compiling the model
lenet_model.compile(
    optimizer=Adam(learning_rate=Configuration["LearningRate"]),
    loss=CategoricalCrossentropy(),
    metrics=[CategoricalAccuracy(), TopKCategoricalAccuracy(k=3)]
)

# Set up Early Stopping and Model Checkpoint
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Ensure checkpoint path exists
checkpoint_path = "/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/ERSFinal/Main/Data/checkpoint.keras"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

# Early Stopping to stop training if validation loss doesn't improve
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Model Checkpoint to save the best model during training
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

steps_per_epoch = 23229 // Configuration["BatchSize"]  # Adjust steps per epoch

train_again = False   # Set to True to train, False to load from file

if not train_again and os.path.exists('EmotionRecognitionSystemFinal1.h5'):
    model = tf.keras.models.load_model('EmotionRecognitionSystemFinal1.h5')
    print("Model loaded from file.")
    history = None  # Set history to None if you're loading the model (no training occurred)
else:
    print("Training the model...")
    history = lenet_model.fit(
        TrainingDS,
        epochs=Configuration["NoOfEpochs"],
        validation_data=ValidationDS,
        batch_size=Configuration["BatchSize"],
        steps_per_epoch=steps_per_epoch,
        callbacks=[early_stopping, checkpoint]
    )
    lenet_model.save('EmotionRecognitionSystemFinal1.h5')
    print("Model trained and weights saved.")

# Step 2: Plot Training Accuracy and Loss
if history is not None:  # Only plot if history exists
    # Accessing keys in the history dictionary
    print(history.history.keys())

    # Plot training and validation metrics
    plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['top_k_categorical_accuracy'], label='Training Top-3 Accuracy')
    plt.plot(history.history['val_top_k_categorical_accuracy'], label='Validation Top-3 Accuracy')

    # Add labels and legend
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Step 3: F1 Score and Confusion Matrix
# Get predictions
y_true = []
y_pred = []
for images, labels in TestingDS:
    predictions = lenet_model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

# Compute F1 Score
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ClassNames)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
