import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FFBlock(nn.Module):
    """
    Trainable block unit for forward-forward training.

    Args:
        block: block to be wrapped.
        lr: learning rate to be used when training this block.
        opt_name: optimizer name to be used when training this block.
        threshold: threshold to be used in block-wise loss function.
        activation: activation function to be used after the block.
    """
    def __init__(self, block: nn.Module, lr: float, opt_name: str, threshold: float, activation: str = "relu"):
        super().__init__()
        self.block = block
        self.threshold = threshold

        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer {opt_name} is not supported.")

        if activation is None:
            self.activation = nn.Identity()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Activation {activation} is not supported.")

    def forward(self, x):
        return self.activation(self.block(self.normalize(x)))

    def train_block(self, x_pos, x_neg) -> None:
        g_pos = self.forward(x_pos).pow(2).view(x_pos.size(0), -1).mean(dim=1)
        g_neg = self.forward(x_neg).pow(2).view(x_neg.size(0), -1).mean(dim=1)
        # The following loss pushes pos (neg) samples to
        # values larger (smaller) than the self.threshold.
        g = torch.cat([-g_pos + self.threshold, g_neg - self.threshold])
        loss = torch.where(g <= 0, F.softplus(g), g + F.softplus(-g)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def normalize(self, x):
        x_norm = x.view(x.size(0), -1).norm(2, dim=1)
        for _ in range(x.dim() - 1):
            x_norm = x_norm.unsqueeze(-1)
        # Avoid division by zero.
        return x / (x_norm + 1e-4)


class LabelEmbedder:
    """
    Class to embed label information into input data.

    Args:
        train_loader: 'DataLoader' for training
        num_classes: number of target classes
        method:
            "top-left": Top left 10 pixels are used to describe label information (original paper).
            "average-subtraction": Subtract the average image for the specified class.
            "average-mixture": Mix the original input with the average image for the specified class.
            "average-2channel": Add the average image for the specified class in the input channel.
            "fixed-example-mixture": Mix the original input with the fixed example for the specified class.
            "fixed-example-2channel": Add the fixed example for the specified class in the input channel.
    """
    def __init__(self, train_loader: DataLoader, num_classes: int = 10, method: str = "top-left", device: str = "None"):
        self.num_classes = num_classes
        self.method = method
        if method.startswith("average"):
            self.average_image_for_each_class = self._average_image_for_each_class(train_loader, device)
        if method.startswith("fixed-example"):
            self.fixed_example_for_each_class = self._fixed_example_for_each_class(train_loader, device)

    def embed_label(self, x: torch.Tensor, y: torch.Tensor | int) -> torch.Tensor:
        """
        Embed label information into input data.

        Args:
            x: image (For MNIST with batch-size 64, x's shape is [64, 1, 28, 28].)
            y: label (tensor with the shape [x.shape[0]] or int)
        Returns:
            Tensor with the shape of [batch_size, num_channels, height, width].
            If `self.method` ends with `2channel`, then the shape gets [batch_size, 2 * in_channel, height, width].
        """
        if self.method == "top-left":
            x_ = x.clone()
            width, height = x.shape[2], x.shape[3]
            x_ = x_.view(x.shape[0], x.shape[1], -1)
            x_[:, :, : self.num_classes] *= 0.0
            x_[range(x.shape[0]), :, y] = x.max()
            return x_.view(x.shape[0], x.shape[1], width, height)
        elif self.method == "average-subtraction":
            return x - self.average_image_for_each_class[y].view(-1, x.shape[1], x.shape[2], x.shape[3])
        elif self.method == "average-mixture":
            return 0.5 * x + 0.5 * self.average_image_for_each_class[y].view(-1, x.shape[1], x.shape[2], x.shape[3])
        elif self.method == "average-2channel":
            return torch.cat(
                (
                    x,
                    torch.broadcast_to(
                        self.average_image_for_each_class[y].view(-1, x.shape[1], x.shape[2], x.shape[3]), x.shape
                    ),
                ),
                dim=1,
            )
        elif self.method == "fixed-example-subtraction":
            return x - self.fixed_example_for_each_class[y].view(-1, x.shape[1], x.shape[2], x.shape[3])
        elif self.method == "fixed-example-mixture":
            return 0.5 * x + 0.5 * self.fixed_example_for_each_class[y].view(-1, x.shape[1], x.shape[2], x.shape[3])
        elif self.method == "fixed-example-2channel":
            return torch.cat(
                (
                    x,
                    torch.broadcast_to(
                        self.fixed_example_for_each_class[y].view(-1, x.shape[1], x.shape[2], x.shape[3]), x.shape
                    ),
                ),
                dim=1,
            )
        else:
            raise ValueError("label_embedding_method: {} is not supported.".format(self.method))

    def _average_image_for_each_class(self, train_loader: DataLoader, device) -> torch.Tensor:
        """
        Calculate the average image for each class.
        Returns:
            Tensor with the shape of [num_classes, num_channels, height, width].
        """
        # Takes some time to compute.
        # It is easier to take averages using 'dataloader.dataset.data',
        # but it returns the averages images for unnormalized data (not processed `torchvision.transforms`).
        sum_image_for_each_class = {}
        for x, y in train_loader:
            for i in range(self.num_classes):
                sum_image_within_class_for_batch = torch.sum(x[y == i], axis=0)
                if i in sum_image_for_each_class:
                    assert (sum_image_for_each_class[i].shape == sum_image_within_class_for_batch.shape)
                    sum_image_for_each_class[i] += sum_image_within_class_for_batch
                else:
                    sum_image_for_each_class[i] = sum_image_within_class_for_batch

        average_image_for_each_class = []
        for i in range(self.num_classes):
            targets = train_loader.dataset.targets
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)
            num_data_within_class = targets[targets == i].numel()
            average_image_for_each_class.append(sum_image_for_each_class[i] / num_data_within_class)

        return torch.stack(average_image_for_each_class, dim=0).to(device)

    def _fixed_example_for_each_class(self, train_loader: DataLoader, device) -> torch.Tensor:
        """
        Fetch a fixed example for each class.
        Returns:
            Tensor with the shape of [num_classes, num_channels, height, width].
        """
        fixed_example_for_each_class = []
        for i in range(self.num_classes):
            for x, y in train_loader:
                if i in y:
                    fixed_example_for_each_class.append(x[y == i][0])
                    break
        return torch.stack(fixed_example_for_each_class, dim=0).to(device)
