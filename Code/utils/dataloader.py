import os
from medmnist import BreastMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

root_datasets = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../Datasets")
)


def load_breastmnist_numpy(root_dir=root_datasets):
    """
    Loads BreastMNIST into NumPy arrays.
    Used when training SVM (Model A).
    """
    os.makedirs(root_dir, exist_ok=True)

    train = BreastMNIST(split="train", root=root_dir, download=False)
    val = BreastMNIST(split="val", root=root_dir, download=False)
    test = BreastMNIST(split="test", root=root_dir, download=False)

    x_train, y_train = train.imgs, train.labels
    x_val, y_val = val.imgs, val.labels
    x_test, y_test = test.imgs, test.labels

    return (x_train, y_train.ravel()), (x_val, y_val.ravel()), (x_test, y_test.ravel())


def load_breastmnist_torch(
    batch_size=64, root_dir=root_datasets, extra_transforms=None
):
    """
    Loads BreastMNIST as PyTorch datasets and dataloaders.
    Used for CNN (Model B).

    Args:
        extra_transforms: list of torchvision transforms (for augmentation)
    """
    os.makedirs(root_dir, exist_ok=True)

    pil_transforms = []
    tensor_transforms = []

    if extra_transforms:
        pil_transforms = extra_transforms.get("pil", [])
        tensor_transforms = extra_transforms.get("tensor", [])

    transform = transforms.Compose(
        pil_transforms
        + [
            transforms.ToTensor(),
        ]
        + tensor_transforms
        + [
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    basic_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    train = BreastMNIST(
        split="train", root=root_dir, transform=transform, download=False
    )
    val = BreastMNIST(
        split="val", root=root_dir, transform=basic_transform, download=False
    )
    test = BreastMNIST(
        split="test", root=root_dir, transform=basic_transform, download=False
    )

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
