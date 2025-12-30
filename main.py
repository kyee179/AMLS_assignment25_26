import argparse
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "Code"))

from Code.utils.dataloader import load_breastmnist_numpy, load_breastmnist_torch
from Code.utils.preprocess import (
    ImagePreprocessor,
    NumpyAugmentations,
    get_torch_augmentations,
)
from Code.A.model_a import ModelA
from Code.B.model_b import ModelB


def plot_learning_curves(history):
    """
    Plots Train vs Val Loss to visualize Overfitting.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Training Loss")
    plt.plot(epochs, history["val_loss"], "r-", label="Validation Loss")
    plt.title("Loss Curve (Overfitting Check)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "b-", label="Training Acc")
    plt.plot(epochs, history["val_acc"], "r-", label="Validation Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()


def run_model_a(augment=False, perform_grid_search=False):
    print("\n=== Running Model A (SVM + HOG + PCA) ===")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_breastmnist_numpy()

    if augment:
        x_train = NumpyAugmentations.add_gaussian_noise(x_train)
        x_train = NumpyAugmentations.adjust_contrast(x_train)

    preprocessor = ImagePreprocessor(use_hog=True, use_pca=True, pca_components=30)
    x_train_feats = preprocessor.fit_transform(x_train)
    x_test_feats = preprocessor.transform(x_test)

    model = ModelA()

    if perform_grid_search:
        results = model.grid_search(x_train_feats, y_train)
        print("\nTop 5 Configs by Accuracy:")
        print(
            results[["param_C", "param_kernel", "mean_test_score", "std_test_score"]]
            .sort_values(by="mean_test_score", ascending=False)
            .head()
        )
    else:
        model.train(x_train_feats, y_train)

    model.evaluate(x_test_feats, y_test)


def run_model_b(augment=False):
    print("\n=== Running Model B (ResNet18) ===")

    extra_transforms = get_torch_augmentations() if augment else None

    train_loader, val_loader, test_loader = load_breastmnist_torch(
        batch_size=32, extra_transforms=extra_transforms
    )

    model = ModelB(epochs=20, learning_rate=0.001)

    history = model.train(train_loader, val_loader)

    plot_learning_curves(history)

    model.evaluate(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["A", "B"], required=True)
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="(Model A Only) Run hyperparameter search",
    )

    args = parser.parse_args()

    if args.model == "A":
        run_model_a(augment=args.augment, perform_grid_search=args.grid_search)
    elif args.model == "B":
        run_model_b(augment=args.augment)
