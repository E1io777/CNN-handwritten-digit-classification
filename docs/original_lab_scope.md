# Original laboratory scope and portfolio extensions

## Original exam material currently available

The supplied exam page is titled **“Lab 1: Constructing a Convolutional Neural Network Architecture for Handwritten Digits Classification and Image Matching.”**

It specifies Python 3.5 or above, NumPy, Matplotlib, TensorFlow, and the MNIST handwritten dataset. The visible learning tasks are:

1. Import modules and set a random seed.
2. Load MNIST and normalize pixel intensities to [0, 1].
3. Check the training data size: 60,000 images at 28 × 28 pixels.
4. Display the pixel values of the first training image.
5. Inspect the structure of training labels.
6. Show the first training label, specified as digit 5.
7. Check the test data size: 10,000 images at the same resolution.
8. Check the test-label array has 10,000 elements.
9. Show the first ten test images and their labels.

The Python program supplied with the exam implements these data-loading, normalization, inspection, and visualization steps.

## What is not yet verified

The page provided so far does not show the full CNN architecture/training requirements or define the image-matching method. Do not claim that image matching was implemented until the remaining exam pages or original program confirm the method.

## Portfolio extensions in this repository

- `src/01_mnist_exploration.py` is a cleaned and commented version of the supplied data-exploration answer.
- `src/02_cnn_classifier.py` is a separate portfolio extension that trains and evaluates a CNN classifier and saves plots. Do not describe it as part of the original exam submission unless the original materials establish that fact.

## Reporting results honestly

Run the CNN script in your own environment before reporting accuracy, loss, training time, or other metrics. Results may vary by software version and hardware. Keep generated results and describe the environment used.
