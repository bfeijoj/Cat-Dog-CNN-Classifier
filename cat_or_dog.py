import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------- Plot Function --------------------------------------------------------------------------------------------

def plot_function(images_array, probabilities = False):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5,len(images_arr) * 3))
    if probabilities is False:
      for img, ax in zip( images_arr, axes):
          ax.imshow(img)
          ax.axis('off')
    else:
      for img, probability, ax in zip( images_arr, probabilities, axes):
          ax.imshow(img)
          ax.axis('off')
          if probability > 0.5:
              ax.set_title("%.2f" % (probability*100) + "% dog")
          else:
              ax.set_title("%.2f" % ((1-probability)*100) + "% cat")
    plt.show()


# -------------------------------------------------------------------------------------------------- Get files ---------------------------------------------------------------------------------------------

!wget https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip

!unzip cats_and_dogs.zip

path = 'cats_and_dogs'

train_dir = os.path.join(path, 'train')
validation_dir = os.path.join(path, 'validation')
test_dir = os.path.join(path, 'test')

total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))

# -------------------------------------------------------------------------------------- Pre processing and training variables -----------------------------------------------------------------------------

batch_size = 128
epochs = 15
image_height = 150
image_width = 150

# ----------------------------------------------------------------------------------------------- New images generator -------------------------------------------------------------------------------------

train_image_generator = ImageDataGenerator(rescale = 1./255)
validation_image_generator = ImageDataGenerator(rescale = 1./255)
test_image_generator = ImageDataGenerator(rescale = 1./255)

train_data_gen = train_image_generator.flow_from_directory(
        train_dir,
        target_size = (image_height, image_width),
        batch_size = batch_size,
        class_mode = 'binary',
        classes = ['cats', 'dogs'])

val_data_gen = validation_image_generator.flow_from_directory(
        validation_dir,
        target_size = (image_height, image_width),
        batch_size = batch_size,
        class_mode = 'binary',
        classes = ['cats', 'dogs'])

test_data_gen = test_image_generator.flow_from_directory(
        PATH,
        classes = ['test'],
        target_size = (image_height, image_width),
        batch_size = batch_size,
        shuffle = False,
        class_mode = None)

train_image_generator = ImageDataGenerator(
rescale = 1./255,
rotation_range = 40,
width_shift_range = 0.2,
height_shift_range = 0.2,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True,
fill_mode = 'nearest')

train_data_gen = train_image_generator.flow_from_directory(batch_size = batch_size,
                                                     directory = train_dir,
                                                     target_size = (image_height, image_width),
                                                     class_mode = 'binary',
                                                     classes = ['cats', 'dogs'])

# --------------------------------------------------------------------------------------------------- CNN Architecture ------------------------------------------------------------------------------------

base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet")
base_model.trainable = False

model = Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(1)
])

base_learning_rate = 0.0001

model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = base_learning_rate),
              loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
              metrics = ['accuracy'])

model.summary()

# -------------------------------------------------------------------------------------------------------- Training ---------------------------------------------------------------------------------------

history = model.fit(train_data_gen,
                    epochs = epochs,
                    steps_per_epoch = total_train // batch_size,
                    validation_data = val_data_gen,
                    validation_steps = total_val // batch_size)

accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, validation_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('cat_or_dog.h5')

# -------------------------------------------------------------------------------------------------------- Predicting -------------------------------------------------------------------------------------

new_model = tf.keras.models.load_model('cat_or_dog.h5')

predictions = new_model.predict(test_data_gen)

sample_test_images = [test_data_gen[0][ii][:] for ii in range(50)]

predictions = (predictions - min(predictions)) / (max(predictions) - min(predictions))

plot_function(sample_test_images, probabilities = predictions)