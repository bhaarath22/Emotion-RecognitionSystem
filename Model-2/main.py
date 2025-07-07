import os
import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, classification_report

# Data preparation
train = ImageDataGenerator(
    rescale=1/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_DS = train.flow_from_directory(
    '/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/ERNDA3Emotions/HASsplittedDS/train',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical'
)

validation_DS = validation.flow_from_directory(
    '/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/ERNDA3Emotions/HASsplittedDS/val',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical'
)

test_DS = test.flow_from_directory(
    '/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/ERNDA3Emotions/HASsplittedDS/test',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Keep labels in order for evaluation
)

# EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Model Definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(200, 200, 3)),

    # First convolutional block
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(2, 2),

    # Second convolutional block
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(0.3),  # Dropout to reduce overfitting

    # Third convolutional block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(0.3),

    # Fourth convolutional block
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(0.3),

    # Flatten and Dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Learning rate scheduler callback
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# Transfer Learning
# we can use a pre-trained model
base_model = tf.keras.applications.VGG16(input_shape=(200, 200, 3), include_top=False, weights='/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/ERNDA3Emotions/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
base_model.trainable = False  # Freeze the base model


transfer_learning_model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the transfer learning model
transfer_learning_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

#model.summary()
# Model training
train_again = False  #  True to train, False to load from file
model_fit = None

if not train_again and os.path.exists('EmotionRecognitionSystem.h5'):
    transfer_learning_model = tf.keras.models.load_model('EmotionRecognitionSystem.h5')
    print("Model loaded from file.")
else:
    print("Training the model...")
    model_fit = transfer_learning_model.fit(  # Training of the transfer learning model
        train_DS,
        steps_per_epoch=train_DS.samples // train_DS.batch_size,
        epochs=30,
        validation_data=validation_DS,
        batch_size=32,
        callbacks=[early_stopping, lr_scheduler]  # Including both callbacks
    )
    transfer_learning_model.save('EmotionRecognitionSystem.h5')
    print("Model trained and weights saved.")


# Evaluation of the model on the test data
test_loss, test_accuracy = transfer_learning_model.evaluate(test_DS)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Calculating F1-score on the test data
test_labels = test_DS.classes
test_predictions = np.argmax(transfer_learning_model.predict(test_DS), axis=1)

f1 = f1_score(test_labels, test_predictions, average='weighted')
print(f"Test F1-Score: {f1:.4f}")

print("Classification Report:")
print(classification_report(test_labels, test_predictions, target_names=test_DS.class_indices.keys()))

# Plotting graphs
plt.figure(figsize=(18, 5))

# Loss graph
if model_fit is not None:
 plt.subplot(1, 3, 1)
 plt.plot(model_fit.history['loss'], label='Training Loss')
 plt.plot(model_fit.history['val_loss'], label='Validation Loss')
 plt.title('Loss During Training and Validation')
 plt.xlabel('Epochs')
 plt.ylabel('Loss')
 plt.legend()

# Accuracy graph
 plt.subplot(1, 3, 2)
 plt.plot(model_fit.history['accuracy'], label='Training Accuracy')
 plt.plot(model_fit.history['val_accuracy'], label='Validation Accuracy')
 plt.title('Accuracy During Training and Validation')
 plt.xlabel('Epochs')
 plt.ylabel('Accuracy')
 plt.legend()

plt.show()
