import os
from medmnist import BreastMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

root_datasets = os.path.abspath("../Datasets")


def load_breastmnist_numpy(root_dir=root_datasets):
    """
    Loads BreastMNIST into NumPy arrays.
    Used when training SVM.
    """
    train = BreastMNIST(split="train", root=root_dir, download=False)
    val = BreastMNIST(split="val", root=root_dir, download=False)
    test = BreastMNIST(split="test", root=root_dir, download=False)

    x_train, y_train = train.imgs, train.labels
    x_val, y_val = val.imgs, val.labels
    x_test, y_test = test.imgs, test.labels

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_breastmnist_torch(batch_size=64, root_dir=root_datasets):
    """
    Loads BreastMNIST as PyTorch datasets and dataloaders.
    Used for CNN.
    """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    train = BreastMNIST(
        split="train", root=root_dir, transform=transform, download=False
    )
    val = BreastMNIST(split="val", root=root_dir, transform=transform, download=False)
    test = BreastMNIST(split="test", root=root_dir, transform=transform, download=False)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
