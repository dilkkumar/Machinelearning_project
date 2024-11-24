# Importing libraries
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Image data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
dataset_path = 'face_emotion_dataset'

# Count the number of subdirectories (classes) in the dataset path
number_of_classes = len(os.listdir(dataset_path))

# Training set (use the entire dataset for training)
training_set = train_datagen.flow_from_directory(
    directory=dataset_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset='training'  # Use training subset
)

# Validation set
validation_set = train_datagen.flow_from_directory(
    directory=dataset_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset='validation'  # Use validation subset
)

# Building the CNN model
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=number_of_classes, activation='softmax')
])

# Compiling the model
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
cnn.fit(x=training_set, epochs=25, validation_data=validation_set)

# Save the trained model
cnn.save('face_emotion_detection_model.h5')
