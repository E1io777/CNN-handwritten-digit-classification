"""Explore the MNIST dataset for the original laboratory task.

This script covers data loading, normalization, shape/label inspection, and
visualization. It does not construct or train a CNN.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

def main():
    np.random.seed(79)
    tf.random.set_seed(79)

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    print("Training image shape:", train_images.shape)
    print("Training label shape:", train_labels.shape)
    print("First training label:", train_labels[0])
    print("\\nPixel values of the first training image:")
    print(train_images[0])

    print("\\nTest image shape:", test_images.shape)
    print("Test label shape:", test_labels.shape)
    print("First 10 test labels:", test_labels[:10])

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for index, axis in enumerate(axes.flat):
        axis.imshow(test_images[index], cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"Label: {test_labels[index]}")
        axis.axis("off")

    fig.suptitle("First 10 MNIST Test Images")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
