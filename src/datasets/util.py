import copy
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def train_val_split(
    train_dataset: torch.utils.data.Dataset, validation_ratio: float = 0.1, seed: int = 1
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Split train dataset into train and validation dataset.
    """
    x_train, x_val, y_train, y_val = train_test_split(
        train_dataset.data, train_dataset.targets, test_size=validation_ratio, random_state=seed
    )

    val_dataset = copy.deepcopy(train_dataset)

    train_dataset.data = x_train
    train_dataset.targets = y_train

    val_dataset.data = x_val
    val_dataset.targets = y_val

    return train_dataset, val_dataset


class TwoCropTransform:
    """
    Creates two crops of the same image.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        return [self.transform(img), self.transform(img)]


class PairedDataset(Dataset):
    """
    `torch.utils.data.Dataset` that returns a pair of two examples that belong to the same class.
    Note that `transform` has already been applied to the examples when creating the argument `dataset`.

    Args:
        dataset: The original dataset such as `torchvision.datasets.MNIST`.
        label_embedding_method: The way to embed label information to data.
            'fixed-example': The other examples are always the same for each class.
            'random-example': The other examples are randomly sampled for each class.
    """

    def __init__(self, dataset: Dataset, label_embedding_method: str):
        self.dataset = dataset
        self.num_classes = len(torch.unique(torch.tensor(dataset.targets)))
        self.class_indices = self._get_class_indices()
        self.label_embedding_method = label_embedding_method

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data1, target = self.dataset[index]

        if self.label_embedding_method == "fixed-example":
            index2 = self.class_indices[target][0]
            data2, target2 = self.dataset[index2]
            assert target == target2
        elif self.label_embedding_method == "random-example":
            while True:
                index2 = random.sample(self.class_indices[target], 1)[0]
                if index != index2:
                    break
            data2, target2 = self.dataset[index2]
            assert target == target2
        elif self.label_embedding_method == "top-left":
            data2 = torch.zeros_like(data1)
            width, height = data1.shape[1], data1.shape[2]
            data2 = data2.view(data1.shape[0], -1)
            data2[:, target] = data1.max()
            data2 = data2.view(data1.shape[0], width, height)
        else:
            assert False, "Unreachable"

        return (data1, data2), target

    def _get_class_indices(self):
        """
        Returns:
            Dictionary mapping each class to a list of indices.
        """
        class_indices = {}
        for i, (_, target) in enumerate(self.dataset):
            if target not in class_indices:
                class_indices[target] = []
            class_indices[target].append(i)
        return class_indices

    def fetch_example_from_class(self, target: int) -> Tensor:
        """
        Returns:
            A example from the class 'target'. The way to fetch depends on the argument `label_embedding_method`.
        """
        if self.label_embedding_method == "fixed-example":
            index = self.class_indices[target][0]
            data1, target1 = self.dataset[index]
            assert target == target1
        elif self.label_embedding_method == "random-example":
            index = random.sample(self.class_indices[target], 1)[0]
            data1, target1 = self.dataset[index]
            assert target == target1
        elif self.label_embedding_method == "top-left":
            _data, _ = self.dataset[0]
            data1 = torch.zeros_like(_data)
            width, height = data1.shape[1], data1.shape[2]
            data1 = data1.view(data1.shape[0], -1)
            data1[:, target] = _data.max()
            data1 = data1.view(data1.shape[0], width, height)
        else:
            assert False, "Unreachable"

        return data1
