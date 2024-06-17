import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# !pip install tensorflow-datasets
import tensorflow_datasets as tfds
import pathlib
from calculate_histogram import calculate

VAIDATION_SPLIT = 0.2

SEED = 123

DATASET_IMAGES = pathlib.Path('datasets/PCB_DATASET/images')

# MY

def custom_mapping_function(image, label):
    def calc(v):
        return calculate(v, 16)
    res = calc(image)
    print(res)
    return res, label

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_IMAGES,
    validation_split=VAIDATION_SPLIT,
    subset="training",
    seed=SEED,
    # image_size=(img_height, img_width),
    batch_size=batch_size
).map(custom_mapping_function)


val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_IMAGES,
    validation_split=VAIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    # image_size=(img_height, img_width),
    batch_size=batch_size
).map(custom_mapping_function)

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# MY

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
