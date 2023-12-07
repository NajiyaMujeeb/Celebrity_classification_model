# Celebrity Image Classification using Convolutional Neural Networks (CNN)

## Overview

This project aims to classify images of five distinct celebrities: Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli. The model is designed to accurately differentiate between these celebrity images using Convolutional Neural Networks (CNN).

## Dataset

The dataset consists of images of the following celebrities:
- Lionel Messi: 36 images
- Maria Sharapova: 34 images
- Roger Federer: 28 images
- Serena Williams: 29 images
- Virat Kohli: 41 images

The images were resized to 128x128 pixels and preprocessed for better training. The dataset was split into 80% for training and 20% for validation purposes.

## Model Architecture

The CNN model architecture used for classification:
- Input Layer: Supports images with sizes of 128x128x3
- Convolutional Layer: (3, 3) kernel with ReLU activation using 32 filters
- Max Pooling Layer: Reduces spatial dimensions for feature extraction
- Flatten Layer: Converts 2D matrix data to a vector
- Dense Layers: Fully connected layers with ReLU activation, one with 0.1 dropout regularization
- Output Layer: Softmax activation for multi-class classification

## Training Details

- Loss Function: Sparse Categorical Crossentropy
- Optimizer: Adam optimizer
- Early Stopping: Used to prevent overfitting by stopping training if no improvement in 10 consecutive rounds
- Epochs: Trained over 25 epochs
- Batch Size: 32

## Results

- Training Accuracy: Improved from an initial 0.24 to a peak of 0.97
- Test Accuracy: Achieved 79.41%

## Code

Below is a code snippet illustrating the model training:

```python
# Python code using TensorFlow/Keras for model training

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(5, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=25, batch_size=32, validation_data=(val_images, val_labels), callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
