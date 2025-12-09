import sys, os

# Add the project root to Python path
project_root = os.path.abspath("..")
sys.path.append(project_root)


from Code.utils.dataloader import load_breastmnist_numpy, load_breastmnist_torch


def main():
    # Load BreastMNIST as NumPy arrays
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_breastmnist_numpy()

    # Load BreastMNIST as PyTorch dataloaders
    train_loader, val_loader, test_loader = load_breastmnist_torch(batch_size=64)


if __name__ == "__main__":
    main()
