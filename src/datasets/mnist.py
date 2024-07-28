from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from src.datasets.util import train_val_split, TwoCropTransform


def get_MNIST_datasets(validation_ratio=0.0, root="~/pytorch_datasets") -> tuple[Dataset, Dataset, Dataset]:
    """
    Returns:
        `Dataset` for normal supervised learning.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False, download=True, transform=transform)

    if validation_ratio > 0:
        train_dataset, val_dataset = train_val_split(train_dataset, validation_ratio)
    else:
        val_dataset = None

    return train_dataset, val_dataset, test_dataset


def get_MNIST_supcon_datasets(validation_ratio=0.0, root="~/pytorch_datasets") -> tuple[Dataset, Dataset, Dataset]:
    """
    Returns:
        `Dataset` for supervised contrastrive learning. Note that `train_dataset` returns a list of size 2 examples by
        data augmentation.
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=28, scale=(0.2, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = MNIST(root=root, train=True, download=True, transform=TwoCropTransform(train_transform))
    test_dataset = MNIST(root=root, train=False, download=True, transform=test_transform)

    if validation_ratio > 0:
        train_dataset, val_dataset = train_val_split(train_dataset, validation_ratio)
        val_dataset.transform = test_transform
    else:
        val_dataset = None

    return train_dataset, val_dataset, test_dataset
