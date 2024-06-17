from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

PCB_IMAGE = '09.JPG'

im = Image.open(PCB_IMAGE)
img_matrix = np.array(im)

def convert_rgb_to_yuv(rgb):
    m = np.array([
        [0.29900],
        [0.58700],
        [0.114001]
    ])
    return np.dot(rgb, m)


def scale(matrix, new_max):
    max_value = np.max(matrix)
    return (matrix / max_value) * new_max


def calculate(img, size=16):
    scaled = scale(convert_rgb_to_yuv(img), size).astype(int)
    flatten = scaled.flatten()
    bincount = np.bincount(flatten, minlength=size)
    res = tf.cast(bincount, tf.float32) / len(flatten)
    return np.reshape(res, (17, 1))

if __name__ == '__main__':
    # Original image

    plt.figure()
    plt.imshow(img_matrix)
    plt.colorbar()
    plt.grid(False)
    plt.show()

    # Converted to YUV

    yuv = convert_rgb_to_yuv(img_matrix)

    plt.figure()
    plt.imshow(yuv)
    plt.colorbar()
    plt.grid(False)
    plt.show()

    # Scaled

    max_pixel_value = 16

    scaled = scale(yuv, max_pixel_value).astype(int)

    plt.figure()
    plt.imshow(scaled)
    plt.colorbar()
    plt.grid(False)
    plt.show()

    # Create histogram

    res = calculate(img_matrix)

    print(res.shape)

    plt.figure()
    plt.hist(res)
    plt.grid(False)
    plt.show()
