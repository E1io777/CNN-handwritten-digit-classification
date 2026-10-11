# MNIST Handwritten Digit Classification

This is a small Python project. 

**“Constructing a Convolutional Neural Network Architecture for Handwritten Digits Classification and Image Matching.”**

## What the program does

- Loads the MNIST handwritten-digit dataset.
- Normalizes image pixel values from 0–255 to 0–1.
- Checks the sizes of the training and test datasets.
- Prints the first training image's pixel values and label.
- Displays the first 10 test images with their labels.

---

## Requirement 

- Python 3
- NumPy
- Matplotlib
- TensorFlow

---
## How to run it

1. Download or clone this repository.
2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the program:

   ```bash
   python main.py
   ```

The first run may need an internet connection so TensorFlow can download MNIST.

---
## Expected dataset sizes

- Training images: 60,000 images, each 28 × 28 pixels.
- Training labels: 60,000 labels.
- Test images: 10,000 images, each 28 × 28 pixels.
- Test labels: 10,000 labels.

This is a learning project. The code and documentation will be improved as I review the remaining requirements.
