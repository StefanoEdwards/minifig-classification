import os
import math
import random
import shutil
import warnings

# Suppress TensorFlow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR
# Ignore the specific PyDataset warning from Keras
warnings.filterwarnings('ignore', message='Your `PyDataset` class should call')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import tensorflow as tf

# Dataset info
BASE_DIR = 'lego/star-wars-images/'
# Class names for classification
names = ["YODA", "LUKE SKYWALKER", "R2-D2", "MACE WINDU", "GENERAL GRIEVOUS"]

# Set random seed for reproducibility
tf.random.set_seed(1)

'''
# This commented section contains code to organize the dataset into train/val/test splits
# Only run this once when setting up your dataset

# Reorganize the folder structure:
if not os.path.isdir(BASE_DIR + 'train/'):
    for name in names:
        os.makedirs(BASE_DIR + 'train/' + name)
        os.makedirs(BASE_DIR + 'val/' + name)
        os.makedirs(BASE_DIR + 'test/' + name)

# Move the image files with a 60/25/15 split for train/val/test
orig_folders = ["0001/", "0002/", "0003/", "0004/", "0005/"]
for folder_idx, folder in enumerate(orig_folders):
    files = os.listdir(BASE_DIR + folder)
    number_of_images = len([name for name in files])
    n_train = int((number_of_images * 0.6) + 0.5)  # 60% for training
    n_valid = int((number_of_images*0.25) + 0.5)   # 25% for validation
    n_test = number_of_images - n_train - n_valid  # Rest for testing
    print(number_of_images, n_train, n_valid, n_test)
    for idx, file in enumerate(files):
        file_name = BASE_DIR + folder + file
        if idx < n_train:
            shutil.move(file_name, BASE_DIR + "train/" + names[folder_idx])
        elif idx < n_train + n_valid:
            shutil.move(file_name, BASE_DIR + "val/" + names[folder_idx])
        else:
            shutil.move(file_name, BASE_DIR + "test/" + names[folder_idx])
'''

# Setup data augmentation for training data to artificially increase dataset size
# Data augmentation helps prevent overfitting by creating variations of training images -> commented out currently
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) # Normalize pixel values to 0,1
''' rotation_range=20,  # Randomly rotate images by up to 20 degrees
    horizontal_flip=True,  # Randomly flip images horizontally
    width_shift_range=0.2,  # Randomly shift images horizontally by up to 20%
    height_shift_range=0.2, # Randomly shift images vertically by up to 20%
    shear_range=0.2,    # Shear intensity
    zoom_range=0.2      # Range for random zoom
)'''

valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Create data generators that load images from directories
# flow_from_directory automatically maps subdirectory names to class labels
train_batches = train_gen.flow_from_directory(
    'lego/star-wars-images/train', # Directory containing training images
    target_size=(256, 256),        # Resize all images to 256x256
    class_mode='sparse',           # Use sparse labels (integers) for categorical classes
    batch_size=4,                  # Process 4 images per batch
    shuffle=True,                  # Shuffle images
    color_mode="rgb",              # Use RGB color channels
    classes=names                  # Class names to use
)

# Validation data generator (shuffle=False for reproducible validation)
val_batches = valid_gen.flow_from_directory(
    'lego/star-wars-images/val',
    target_size=(256, 256),
    class_mode='sparse',
    batch_size=4,
    shuffle=False,
    color_mode="rgb",
    classes=names
)

# Test data generator
test_batches = test_gen.flow_from_directory(
    'lego/star-wars-images/test',
    target_size=(256, 256),
    class_mode='sparse',
    batch_size=4,
    shuffle=False,
    color_mode="rgb",
    classes=names
)

# Function to visualize a batch of images with their labels and optional predictions
def show(batch, pred_labels=None):
    plt.figure(figsize=(10,10))
    for i in range(4):  # Show 4 images in a 2x2 grid
        plt.subplot(2,2,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(batch[0][i], cmap=plt.cm.binary)  # Display the image
        # Get the true label for this image
        lbl = names[int(batch[1][i])]
        # If predictions are provided, add them to the label
        if pred_labels is not None:
            lbl += "/ Pred:" + names[int(pred_labels[i])]
        plt.xlabel(lbl)  # Display the label under the image
    plt.show()

'''
# Commented visualization code
show(test_batch)
show(train_batch)
'''

# Create a CNN model for image classification
model = tf.keras.models.Sequential([
    # Input layer with 256x256 RGB images
    tf.keras.layers.Input(shape=(256, 256, 3)),
    # First convolutional block
    tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),  # Reduce spatial dimensions by half
    # Second convolutional block
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),  # Further reduce spatial dimensions
    # Flatten the 3D feature maps to 1D feature vectors
    tf.keras.layers.Flatten(),
    # Dense hidden layer with ReLU activation
    tf.keras.layers.Dense(64, activation='relu'),
    # Output layer with 5 units (one per class), no activation (logits)
    tf.keras.layers.Dense(5)
])

# Display model's architecture
print(model.summary())

# Configure model training params
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # from_logits=True because final layer has no activation
optim = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adam optimizer with default learning rate
metrics = ["accuracy"]  # Track accuracy as our performance metric

# Compile the model with the specified parameters
model.compile(optimizer=optim, loss=loss, metrics=metrics)

# Maximum number of training epochs
epochs = 30

# Early stopping callback to prevent overfitting
# Stop training when validation loss doesn't improve for 5 consecutive epochs
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",  # Monitor validation loss
    patience=5,          # Number of epochs with no improvement before stopping
    verbose=2            # Print a message when training stops early
)

# Train the model, storing training history for later visualization
history = model.fit(
    train_batches,               # Training data
    validation_data=val_batches, # Validation data for monitoring
    callbacks=[early_stopping],  # Use early stopping
    epochs=epochs,               # Maximum number of epochs to train
    verbose=2                    # Level of logging
)

'''
# Commented code to save the trained model
model.save('my_model.keras')
'''

# Visualize training history (loss and accuracy)
plt.figure(figsize=(16, 6))
# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.grid()
plt.legend(fontsize=15)

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='valid acc')
plt.grid()
plt.legend(fontsize=15)

# Evaluate model performance on the test set
model.evaluate(test_batches, verbose=2)

# Make predictions on the test data
predictions = model.predict(test_batches)
# Convert logits to probabilities using softmax
predictions = tf.nn.softmax(predictions)
# Get the predicted class indices (argmax of probabilities)
labels = np.argmax(predictions, axis=1)

# Display the true labels for the first batch of test data
print(test_batches[0][1])
# Display predicted labels for the first batch
print(labels[0:4])
# Visualize the first batch with predictions
show(test_batches[0], labels[0:4])