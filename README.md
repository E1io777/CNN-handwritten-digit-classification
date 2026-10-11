# MNIST Handwritten Digit Data Exploration

This repository contains Python program

**“Constructing a Convolutional Neural Network Architecture for Handwritten Digits Classification and Image Matching.”**

---
## What the program does

- Imports NumPy, Matplotlib, and TensorFlow.
- Sets random seeds.
- Loads the MNIST training and test datasets.
- Normalizes pixel values from 0–255 to 0–1.
- Checks dataset dimensions and label information.
- Displays the first 10 test images with their correct labels.

---
## Example output

The output from my program shows:

- 60,000 training labels.
- The first training label, which is `5`.
- 10,000 test images, each measuring 28 × 28 pixels.
- 10,000 test labels.
- A plot of the first 10 test images and their labels.

---
## Requirements

- Python 3
- NumPy
- Matplotlib
- TensorFlow

---
## How to run

1. Download or clone this repository.
2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the program:

   ```bash
   python main.py
   ```

The first run may need an internet connection so TensorFlow can download the MNIST dataset.

---
## Scope

The code in this repository represents the question and program currently documented here. It loads and explores the dataset; it does **not** train a CNN or implement a separate image-matching method. I will describe only what the program actually does, rather than claim features that are not implemented.
