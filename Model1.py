import os
import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# for loading images from a folder
def loadImages(folder, label_name):
    images = []
    labels = []
    for label in ['happy', 'nothappy']:
        path = os.path.join(folder, label)
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                img = cv2.resize(img, (256, 256))  # Resizing
                images.append(img)
                labels.append(1 if label == 'happy' else 0)  # 1 for happy, 0 for not_happy
    return np.array(images), np.array(labels)

trainingDS = '/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/Accuracy&loss/EmotionsDS/training'
validationDS = '/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/Accuracy&loss/EmotionsDS/validation'
testingDS = '/Users/bharathgoud/PycharmProjects/machineLearing/EmotionRecognition/Accuracy&loss/EmotionsDS/testing'

X_train, y_train = loadImages(trainingDS, 'train')
X_val, y_val = loadImages(validationDS, 'validation')
X_test, y_test = loadImages(testingDS, 'test')

# X_train, X_val, X_test - Arrays of image data (pixels)
# y_train, y_val, y_test- Arrays of labels (0 or 1)

# Normalization of the image pixel values  to 0-1 from 0-255
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# Converting labels to categorical data , happy-1 and nothappy-2
# to_categorical() converts class labels into one-hot encoded vectors.

y_train = to_categorical(y_train, 2)
y_val = to_categorical(y_val, 2)
y_test = to_categorical(y_test, 2)

# Visualizing some images from the trainingDS
# def plotingImages(images, labels, label_names, num_samples=5):
#     plt.figure(figsize=(10, 2))
#     for i in range(num_samples):
#         plt.subplot(1, num_samples, i + 1)
#         plt.imshow(images[i])
#         plt.title(label_names[np.argmax(labels[i])])
#         plt.axis('off')
#     plt.show()
#
# label_names = ['nothappy', 'happy'] #[0,1]
# plotingImages(X_train, y_train, label_names)

# Checking if a trained model already exists with same name
modelPath = 'EMR_NDA.h5'
if os.path.exists(modelPath):
    print("Loading existing model")
    model = tf.keras.models.load_model(modelPath)
else:
    print("Training the  model")
# creating a CNN model
    model = Sequential()
    ''' 1st Convolutional Layer
     This layer is used to extract features from the input images by applying convolution operations.
     It acts as a feature extractor, learning various spatial hierarchies of features from the images, 
    such as edges, textures, shapes, and eventually more complex patterns as more layers are added.'''
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256,256, 3))) #kernel size (3,3)
    '''The number of filters or kernels(32) used in this layer.
     Each filter will learn a different feature of the image'''
    model.add(MaxPooling2D(pool_size=(2, 2)))#layer reduces the spatial dimensions (height and width)
    '''3x3 filter slides across the entire image, computing the dot product between the filter values
     and the input image values at each position. This operation results in a feature map that
      highlights specific features detected by the filter.'''
# 2nd Convolutional Layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
# 3rd Convolutional Layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and Fully Connected Layers
    model.add(Flatten())
    '''converts the 3D feature maps output by the convolutional and pooling layers into a 1D vector.
     This flattening is necessary because fully connected layers expect a 1D input.'''

    '''feature map of size (32, 32, 128), the Flatten layer will convert 
    it into a 1D vector of size 32 * 32 * 128 = 131072.'''

    model.add(Dense(128, activation='relu'))
    '''fully connected neural network layer where every neuron is connected to every neuron in
     the previous and next layers. It helps the model learn from the features extracted by
      the convolutional layers and make decisions about classifying the input.'''
    model.add(Dropout(0.5))
    '''Dropout is a regularization technique used to prevent overfitting.
     It randomly sets a fraction of the input units to zero during training'''
# Output Layer
    model.add(Dense(2, activation='softmax'))
    # 2-->number of neurons corresponds to the number of classes Currently. we are working on only happy or nothappy
    ''' The softmax activation function is used to convert the raw scores (logits) produced by
     the layer into probabilities. The output values range from 0 to 1 and sum to 1, 
     making it suitable for multi-class classification.'''
# Compilation of  the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Training of the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)
# trained model is saved at a path
    model.save(modelPath)
 # Ploting accuracy and loss graph
    plt.figure(figsize=(12, 4))
# Plot training & validation accuracy graph
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
# Ploting training & validation loss graph
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
# Evaluation of the model on the testingDS
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
# Ploting confusion matrix graph for the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Happy', 'Happy'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Loss and Accuracy per Batch using training history
if 'history' in locals():
    # Loss and Accuracy per Batch using training history (this only works if the model was trained here)
    batch_losses = history.history['loss']
    batch_accuracies = history.history['accuracy']

    plt.figure(figsize=(10, 5))
    plt.plot(batch_losses, label='Loss per Batch', color='red')
    plt.plot(batch_accuracies, label='Accuracy per Batch', color='blue')
    plt.title("Loss and Accuracy per Batch")
    plt.xlabel("Batch")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

# this model is overfitting due to less amount of data we have taken
# we can add Data augmentation ,increase the dataset size ,Regularization, Early stoping
# Test Accuracy is  68.18 %
