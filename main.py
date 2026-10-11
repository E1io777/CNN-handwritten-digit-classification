import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Set random seeds
np.random.seed(79)
tf.random.set_seed(79)

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values from 0-255 to 0-1
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Check the training data
image_shape = train_images.shape
print("Training image shape:", image_shape)

pixel_value = train_images[0]
print("Pixel values of the first training image:")
print(pixel_value)

structure_training_labels = train_labels.shape
print("Training label shape:", structure_training_labels)
print("First training label:", train_labels[0])

# Check the test data
print("Test image shape:", test_images.shape)
print("Test label shape:", test_labels.shape)

# Display the first 10 test images and their labels
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[i], cmap="gray")
    plt.title(f"Label: {test_labels[i]}")

plt.tight_layout()
plt.show()
