# Handwritten Digit Classification with a Convolutional Neural Network

**Original university lab:** *Constructing a Convolutional Neural Network Architecture for Handwritten Digits Classification and Image Matching*

This repository documents an undergraduate laboratory exercise using the MNIST handwritten-digit dataset, then develops a clearly separated CNN classification experiment as a portfolio extension.

> **Scope note:** The exam page currently documented here covers loading and exploring MNIST. The remaining exam instructions have not yet been verified, so image matching is not presented as an implemented feature. See [the original lab scope](docs/original_lab_scope.md).

## Project goals

- Load and inspect the MNIST dataset.
- Normalize image pixel intensities to the range [0, 1].
- Inspect labels and visualize example digits.
- Build and evaluate a CNN classifier as a separate extension.
- Record real evaluation outputs and discuss limitations.

## Repository structure

\`\`\`text
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── 01_mnist_exploration.py
│   └── 02_cnn_classifier.py
├── docs/
│   └── original_lab_scope.md
└── results/                 # Generated locally by the scripts
\`\`\`

## Dataset

The scripts use MNIST through tensorflow.keras.datasets.mnist. MNIST contains 60,000 training images and 10,000 test images. Each image is a 28 × 28 grayscale image representing a digit from 0 to 9. TensorFlow downloads the dataset the first time it is loaded, so an internet connection may be required on the first run.

## Setup

Use a virtual environment if possible.

\`\`\`bash
python -m venv .venv
\`\`\`

Activate it, then install the dependencies:

**Windows**
\`\`\`powershell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
\`\`\`

**macOS / Linux**
\`\`\`bash
source .venv/bin/activate
pip install -r requirements.txt
\`\`\`

## Run the original-data exploration

\`\`\`bash
python src/01_mnist_exploration.py
\`\`\`

This script reports the dataset dimensions, displays the pixel values of the first training image, prints its label, and plots the first ten test images with their labels.

Expected dataset dimensions:

| Array | Expected shape |
|---|---|
| Training images | (60000, 28, 28) |
| Training labels | (60000,) |
| Test images | (10000, 28, 28) |
| Test labels | (10000,) |

The first training label in the standard MNIST dataset is 5. Pixel values are converted to floating-point values between 0 and 1 by dividing by 255.

## Run the CNN portfolio extension

\`\`\`bash
python src/02_cnn_classifier.py
\`\`\`

The second script trains a small CNN, evaluates it on the held-out MNIST test set, and writes plots and a confusion matrix to the results/ directory. Training time and results depend on the local environment. **No accuracy result is claimed here in advance; run the script to generate the actual measurements.**

The CNN experiment is labelled as a portfolio extension because the exam page documented so far does not establish that this exact architecture was part of the original submission.

## Image matching

Classification and image matching are different tasks. This project currently documents digit classification; a separate matching method will only be added after the remaining exam instructions or original implementation clarify what matching was required. No image-matching results are claimed at this stage.

## Reproducibility and limitations

- NumPy and TensorFlow random seeds are set in the scripts to improve repeatability.
- Seeds alone do not guarantee bit-for-bit identical results across all TensorFlow versions, devices, or hardware.
- The MNIST test set is used for final evaluation; it should not be used to tune the model repeatedly.
- The model is a learning experiment, not a production handwriting-recognition system.
- Run the scripts in the documented environment and retain generated outputs if you want to report specific metrics.

## Academic context

The project originated as a university laboratory exercise. The original task and the later portfolio extension are documented separately to make the development history clear. Any future changes, experiments, or results should be described accurately and should not be represented as part of the original exam unless supported by the original materials.

## License

No license has been added yet. Add one only after confirming that you have the right to publish the code and any included materials.
