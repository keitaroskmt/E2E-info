from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from src.datasets.util import train_val_split, TwoCropTransform


def get_CIFAR_datasets(
    validation_ratio=0.0, dataset_name="cifar10", seed=1, root="~/pytorch_datasets"
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Returns:
        `Dataset` for normal supervised learning.
    """
    if dataset_name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        Dataset = CIFAR10
    elif dataset_name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        Dataset = CIFAR100
    else:
        raise ValueError("dataset_name must be either cifar10 or cifar100.")

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_dataset = Dataset(root=root, train=True, download=True, transform=train_transform)
    test_dataset = Dataset(root=root, train=False, download=True, transform=test_transform)

    if validation_ratio > 0:
        train_dataset, val_dataset = train_val_split(train_dataset, validation_ratio, seed)
        val_dataset.transform = test_transform
    else:
        val_dataset = None

    return train_dataset, val_dataset, test_dataset


def get_CIFAR_supcon_datasets(
    validation_ratio=0.0, dataset_name="cifar10", seed=1, root="~/pytorch_datasets", train_input_size=28
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Returns:
        `Dataset` for supervised contrastrive learning. Note that `train_dataset` returns a list of size 2 examples by
        data augmentation.
    """
    if dataset_name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        Dataset = CIFAR10
    elif dataset_name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        Dataset = CIFAR100
    else:
        raise ValueError("dataset_name must be either cifar10 or cifar100.")

    assert train_input_size in [28, 32]
    if train_input_size == 32:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=28, scale=(0.2, 1.0)),
                transforms.Resize(32),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=28, scale=(0.2, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_dataset = Dataset(root=root, train=True, download=True, transform=TwoCropTransform(train_transform))
    test_dataset = Dataset(root=root, train=False, download=True, transform=test_transform)

    if validation_ratio > 0:
        train_dataset, val_dataset = train_val_split(train_dataset, validation_ratio, seed)
        val_dataset.transform = test_transform
    else:
        val_dataset = None

    return train_dataset, val_dataset, test_dataset
