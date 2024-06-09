import torch
from torch import nn
from torch.utils.data import DataLoader


def calc_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> float:
    """Calculate the classification accuracy on the given dataset."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            correct += pred.argmax(dim=1).eq(y).sum().item()
    return correct / len(loader.dataset)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
