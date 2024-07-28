import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.forward_forward_block import FFBlock, LabelEmbedder


class FFModel(torch.nn.Module):
    """
    Base class for models trained with forward-forward algorithm.

    Args:
        Layers: list of layers trained with local forward-forward loss.
        num_classes: number of target classes.
        label_embedder: `LabelEmbedder` object, which embeds label information into iput data.
    """

    def __init__(self, layers: nn.ModuleList, num_classes: int, label_embedder: LabelEmbedder):
        super().__init__()
        self.layers = layers
        self.num_classes = num_classes
        self.label_embedder = label_embedder

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Predicted labels with the shape [x.size(0)].
        """
        goodness_per_label = []
        for label in range(self.num_classes):
            h = self.label_embedder.embed_label(x, label)
            goodness = []
            for layer in self.layers:
                # Takes the mean of squares for each data output
                h = layer(h)
                if isinstance(layer, FFBlock):
                    goodness += [h.pow(2).view(h.size(0), -1).mean(dim=1)]
            # List of torch.tensor of size (x.size(0), 1)
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        # torch.tesnor of size (x.size(0), num_classes)
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train_model_simultaneously(self, train_loader: DataLoader, num_epochs: int, device: str):
        """
        Train the model with local Forward-Forward loss.
        Each layer is trained simultaneously for every forward pass.
        """
        torch.autograd.set_detect_anomaly(True)
        for _ in range(num_epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                x_pos = self.label_embedder.embed_label(x, y)
                rnd = torch.randperm(x.size(0))
                x_neg = self.label_embedder.embed_label(x, y[rnd])

                h_pos, h_neg = x_pos, x_neg
                for layer in self.layers:
                    if isinstance(layer, FFBlock):
                        layer.train_block(h_pos, h_neg)
                    h_pos, h_neg = layer(h_pos).detach(), layer(h_neg).detach()

    def train_model_sequentially(self, train_loader: DataLoader, num_epochs: int, device: str):
        """
        Train the model with local Forward-Forward loss.
        Each layer is trained sequentially, starting from the first layer.
        This is a method implemented in the original code.
        """
        torch.autograd.set_detect_anomaly(True)
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, FFBlock):
                continue
            for _ in range(num_epochs):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    x_pos = self.label_embedder.embed_label(x, y)
                    rnd = torch.randperm(x.size(0))
                    x_neg = self.label_embedder.embed_label(x, y[rnd])
                    h_pos, h_neg = x_pos, x_neg
                    # preparing training data for layer i
                    for earlier_layer in self.layers[:i]:
                        h_pos, h_neg = earlier_layer(h_pos).detach(), earlier_layer(h_neg).detach()
                    layer.train_block(h_pos, h_neg)


class FFMLP(FFModel):
    """
    Multi Layer Perceptron trained with forward-forward algorithm.
    Args:
        dims: dimensions of each layer. (e.g. [784, 1000, 1000, 1000, 1000, 10]])
    """

    def __init__(self, lr: float, opt_name: str, threshold: float, num_classes: int, dims: list[int], label_embedder: LabelEmbedder, device="cuda"):
        layers = nn.ModuleList([nn.Flatten()])
        for d in range(len(dims) - 1):
            layers.append(
                FFBlock(
                    nn.Linear(in_features=dims[d], out_features=dims[d + 1], device=device),
                    lr=lr,
                    opt_name=opt_name,
                    threshold=threshold,
                )
            )
        super().__init__(layers, num_classes, label_embedder)


class FFCNN(FFModel):
    """
    Convolutional Neural Network trained with forward-forward algorithm.
    """

    def __init__(self, lr: float, opt_name: str, threshold: float, num_classes: int, image_width: int, label_embedder: LabelEmbedder, num_channels=1, device="cuda"):
        layers = nn.ModuleList([
            FFBlock(
                nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=(3, 3), device=device),
                lr=lr,
                opt_name=opt_name,
                threshold=threshold,
            ),
            FFBlock(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), device=device),
                lr=lr,
                opt_name=opt_name,
                threshold=threshold,
            ),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            FFBlock(
                nn.Linear(in_features=(image_width - 4) ** 2 * 64 // 4, out_features=128, device=device),
                lr=lr,
                opt_name=opt_name,
                threshold=threshold,
            ),
            FFBlock(
                nn.Linear(in_features=128, out_features=num_classes, device=device),
                lr=lr,
                opt_name=opt_name,
                threshold=threshold,
            ),
        ])
        super().__init__(layers, num_classes, label_embedder)
