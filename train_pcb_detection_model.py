# !pip install tensorflow-datasets
import pathlib

import tensorflow as tf

from calculate_histogram import calculate
# example of converting an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img


HISTOGRAM_WIDTH = 16
VALIDATION_SPLIT = 0.05
SEED = 123
DATASET_IMAGES = pathlib.Path('datasets/PCB_DATASET/images')


batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_IMAGES,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=(300, 300),
    batch_size=50
    # batch_size=50
)

# n = len(list(train_ds))

# train_ds = tf.reshape(train_ds, (n, 300, 300, 3))

def convert_to_histogram(dataset, labels):
    print("labels shape")
    # print(labels.shape)
    # modified_image = tf.numpy_function(lambda x: calculate(x), [dataset], tf.float32)
    # modified_image = tf.reshape(modified_image, (HISTOGRAM_WIDTH + 1, 1), name="change_shape")
    return dataset, labels

    # modified_image = tf.numpy_function(calculate, [dataset], tf.float32)
    # return modified_image, labels

print("before")
print(train_ds)

train_ds = train_ds

print("after")
print(train_ds)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_IMAGES,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=(300, 300),
    batch_size=50
    # batch_size=50
)

tf.print(train_ds)

model = tf.keras.Sequential([
    # tf.keras.layers.Input(shape=(HISTOGRAM_WIDTH + 1, 1)),
    tf.keras.layers.Input(shape=(300, 300, 3)),
    tf.keras.layers.Dense(6, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(train_ds, epochs=10)

test_loss, test_acc = model.evaluate(val_ds, verbose=2)

print('\nTest accuracy:', test_acc)
