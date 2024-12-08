# Dogs v Cats Image Classification Project

## Overview
This project aims to develop a deep learning model capable of correctly classifying images of dogs and cats using TensorFlow. The project demonstrates various techniques such as image augmentation, dropout regularization, and transfer learning to improve model performance. Initially, a subset of the original Dogs vs. Cats dataset containing 3000 images is used for demonstration purposes. 

## Dataset
The subset dataset used in this project can be found [here](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip). It consists of 3000 images sampled from the original dataset of 25000 images.

## Libraries and Setup
```python
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O /tmp/cats_and_dogs_filtered.zip

import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Extracting the dataset
local_zip = '/tmp/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

# Directory setup
base_dir = '/tmp/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
```
## Model Architecture
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
## Image Augmentation and Training
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2
)
```
## Results and Analysis
After training the model, the accuracy and loss metrics are visualized using matplotlib. It's observed that the model suffers from overfitting, with high training accuracy (~100%) and lower validation accuracy (~70-75%). To address this, image augmentation is applied, leading to improved performance.

## Transfer Learning with Inception-v3
To further improve performance, transfer learning using the Inception-v3 model is employed. The Inception-v3 model, pre-trained on a large dataset, is fine-tuned for the task of dogs vs. cats classification. With transfer learning, a validation accuracy of 95% is achieved.

## Testing with Custom Images
The trained model is tested with custom images to classify them as either dogs or cats.

## Conclusion
This project showcases various techniques in deep learning, including image augmentation, dropout regularization, and transfer learning, to classify images of dogs and cats. Through experimentation and optimization, significant improvements in model performance are achieved, demonstrating the effectiveness of these techniques in image classification tasks.

