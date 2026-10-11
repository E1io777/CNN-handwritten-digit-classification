"""Train and evaluate a small CNN on MNIST.

This is a portfolio extension, separate from the original data-exploration
exercise. Run it to generate actual metrics and plots.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

SEED = 79
EPOCHS = 5
BATCH_SIZE = 128
RESULTS_DIR = Path("results")

def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(64, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ], name="mnist_cnn")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    model = build_model()
    model.summary()
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, digits=4))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curves.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=np.arange(10),
        cmap="Blues", colorbar=False, ax=axis,
    )
    axis.set_title("MNIST Test-Set Confusion Matrix")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for index, axis in enumerate(axes.flat):
        axis.imshow(x_test[index].squeeze(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"True: {y_test[index]} | Pred: {predictions[index]}")
        axis.axis("off")
    fig.suptitle("Example MNIST Predictions")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sample_predictions.png", dpi=160)
    plt.close(fig)
    print(f"\\nPlots saved in: {RESULTS_DIR.resolve()}")

if __name__ == "__main__":
    main()
