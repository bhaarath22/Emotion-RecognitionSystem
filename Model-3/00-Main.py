
# Standard Libraries
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
# from tensorflow.keras.metrics import F1Score
# Advanced Visualization
import plotly.express as px
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
# Lets Start
ClassNames =['anger','contempt','disgust','fear','happy','neutral','sad','surprise']
ImageSize=256

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
'''
# Set the figure size
plt.figure(figsize=(12, 12))

# Take a batch of images and labels from the Training dataset
for images, labels in TrainingDS.take(1):  # Use take(1) to get one batch take(2) for 2 batches etc..
    for i in range(20):  # Adjust the range
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i] / 255.0)  # Normalize the image values to [0, 1]
        plt.title(ClassNames[tf.argmax(labels[i], axis=0).numpy()])  # Display class name
        plt.axis("off")  # Hide the axes

# Show the plot
plt.show() '''

# Data Augmentation
# Create a sequential model for applying various data augmentations to images
AugmentLayers = Sequential([
RandomRotation(factor=(-0.025, 0.025)), # Rotate images randomly within a range of -2.5 to +2.5 degrees
RandomFlip(mode='horizontal'),  # Randomly flip images horizontally (left to right)
RandomContrast(factor=0.1),# Adjust the contrast of images randomly by up to 10%
RandomBrightness(factor=(0.1, 0.2)), # Adjust brightness randomly between 10% and 20%
RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)) # Randomly translate images by up to 10% of their height and width
])


def AddGaussianNoise(image, mean=0.0, stddev=1.0):
    noise = tf.random.normal(shape=tf.shape(image), mean=mean, stddev=stddev, dtype=image.dtype)
    return image + noise

def ColorJitter(image, brightness=0.1, contrast=0.1):
    """Randomly change brightness and contrast of the image."""
    # Change brightness
    image = tf.image.random_brightness(image, max_delta=brightness)
    # Change contrast
    image = tf.image.random_contrast(image, lower=1-contrast, upper=1+contrast)
    return image
def AugmentLayer(image, label):
    # existing augmentations
    image = AugmentLayers(image, training=True)
    # Gaussian Noise
    image = AddGaussianNoise(image)
    # Convert back to a tensor after adding noise
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    # Apply Color Jitter
    image = ColorJitter(image)
    return image, label


# TrainingDS = TrainingDS.map(AugmentLayer, num_parallel_calls=tf.data.AUTOTUNE)
#
# TrainingDS = TrainingDS.prefetch(tf.data.AUTOTUNE)
# # # Checking the shapes and types of dataset
# for images, labels in trainingDS.take(1):
#      print(images.shape, labels.shape)  # shapes of the images and labels


  #number of samples in the training dataset
# num_samples = TrainingDS.cardinality().numpy()
# print(f"Number of samples in trainingDS: {num_samples}")

ValidationDS = ValidationDS.prefetch(tf.data.AUTOTUNE)
def box(lambda_value, image_size):
    """
    Generate a random bounding box within an image for CutMix augmentation.

    Args:
        lambda_value (float): A value between 0 and 1 used to determine the size of the bounding box.
        image_size (int): The size of the image (height and width).

    Returns:
        Tuple[int, int, int, int]: The y and x coordinates of the top-left corner,
                                    height, and width of the bounding box.
    """
    # Random x and y coordinates from a uniform distribution within the image size
    r_x = tf.cast(tfp.distributions.Uniform(0, image_size).sample(1)[0], dtype=tf.int32)
    r_y = tf.cast(tfp.distributions.Uniform(0, image_size).sample(1)[0], dtype=tf.int32)

    # Calculate the width and height of the box using the provided lambda value
    r_w = tf.cast(image_size * tf.math.sqrt(1 - lambda_value), dtype=tf.int32)
    r_h = tf.cast(image_size * tf.math.sqrt(1 - lambda_value), dtype=tf.int32)

    # Center the box around the randomly sampled coordinates (r_x, r_y)
    r_x = tf.clip_by_value(r_x - r_w // 2, 0, image_size - 1)
    r_y = tf.clip_by_value(r_y - r_h // 2, 0, image_size - 1)

    # Calculate the right and bottom edges of the bounding box
    x_b_r = tf.clip_by_value(r_x + r_w, 0, image_size)
    y_b_r = tf.clip_by_value(r_y + r_h, 0, image_size)

    # Calculate the width and height of the bounding box
    r_w = tf.maximum(x_b_r - r_x, 1)
    r_h = tf.maximum(y_b_r - r_y, 1)

    return r_y, r_x, r_h, r_w

def cutmix(TrainingDS1, TrainingDS2):
    """
    Apply CutMix augmentation to a pair of training datasets.

    Args:
        train_dataset_1: A tuple containing an image and its label.
        train_dataset_2: A tuple containing another image and its label.

    Returns:
        Tuple[Tensor, Tensor]: The augmented image and the mixed label.
    """
    (image_1, label_1), (image_2, label_2) = TrainingDS1, TrainingDS2

    # Sample lambda from the Beta distribution
    lambda_value = tfp.distributions.Beta(0.2, 0.2).sample(1)[0]

    # Generating a random bounding box
    r_y, r_x, r_h, r_w = box(lambda_value, ImageSize)

    # Croping and pading the  images according to the bounding box
    crop_2 = tf.image.crop_to_bounding_box(image_2, r_y, r_x, r_h, r_w)
    pad_2 = tf.image.pad_to_bounding_box(crop_2, r_y, r_x, ImageSize, ImageSize)

    crop_1 = tf.image.crop_to_bounding_box(image_1, r_y, r_x, r_h, r_w)
    pad_1 = tf.image.pad_to_bounding_box(crop_1, r_y, r_x, ImageSize, ImageSize)

    # Mixing the two images
    image = image_1 - pad_1 + pad_2

    # Calculating mixed label
    lambda_value = tf.cast(1 - (r_w * r_h) / (ImageSize * ImageSize), dtype=tf.float32)
    label = lambda_value * tf.cast(label_1, dtype=tf.float32) + (1 - lambda_value) * tf.cast(label_2, dtype=tf.float32)

    return image, label
'''
# training datasets
TrainDS1 = TrainingDS.map(AugmentLayer, num_parallel_calls=tf.data.AUTOTUNE)
TrainDS2 = TrainingDS.map(AugmentLayer, num_parallel_calls=tf.data.AUTOTUNE)

# Creating a mixed dataset using CutMix
mixed_ds = tf.data.Dataset.zip((TrainDS1, TrainDS2))
trainingDS = mixed_ds.map(cutmix, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)'''
'''prefetch method preloads the next batch of data while the current one is being processed
 by the mode. tf.data.AUTOTUNE enables TensorFlow to automatically tune the number 
 of batches to prefetch based on system resources and workload'''

# Shape=Sequential([Resizing(256,256),
#                   Rescaling(1./255)])



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

steps_per_epoch =23229// Configuration["BatchSize"]  # Adjust steps per epoch

train_again = True   # Set to True to train, False to load from file

if not train_again and os.path.exists('EmotionRecognitionSystemFinal1.h5'):
    model = tf.keras.models.load_model('EmotionRecognitionSystemFinal1.h5')
    print("Model loaded from file.")
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



results = lenet_model.evaluate(TestingDS, verbose=0)
loss = results[0]
accuracy = results[1]
top_k_accuracy = results[2]

print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")
print(f"Test top-k accuracy: {top_k_accuracy}")

# Step 2: Plot Training Accuracy and Loss
history_dict = history.history

# Plot accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_dict['accuracy'], label='Training Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history_dict['loss'], label='Training Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
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
