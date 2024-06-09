import dataclasses
from math import ceil

import torch
from torch import nn, Tensor

from models.layer_wise_loss import LayerWiseLossConfig, LayerWiseLoss


class ProjectionHead(nn.Module):
    """
    Projection head attached to the end of each layer.

    Args:
        in_channels: Number of input channels.
        input_size: Size of input image.
        in_featuers: Number of input features. It can be specified instead of `in_channels` and `input_size`.
        out_features: Number of output features..
        head_type: Type of projection head. "linear", "mlp", "conv", "conv_2" or "identity" can be specified.
                "similarity"-type loss uses convolutional head for layer-wise training.
    """

    def __init__(
        self,
        in_channels: int = None,
        input_size: int = None,
        in_features: int = None,
        out_features: int = None,
        head_type: str = "linear",
    ):
        super().__init__()
        self.head_type = head_type
        if head_type == "identity":
            self.head = nn.Identity()
        else:
            if in_channels is not None and input_size is not None and in_features is not None:
                if in_channels * input_size * input_size != in_features:
                    raise ValueError("Invalid input arguments.")
            if in_features is None:
                in_features = in_channels * input_size * input_size

            if head_type == "conv":
                self.head = nn.Conv2d(
                    in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1
                )
            elif head_type == "conv_2":
                self.head = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
                )
            elif head_type == "linear":
                self.head = nn.Linear(in_features=in_features, out_features=out_features)
            elif head_type == "mlp":
                self.head = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=in_features),
                    nn.ReLU(),
                    nn.Linear(in_features=in_features, out_features=out_features),
                )
            else:
                raise ValueError("Invalid head type: {}".format(head_type))

    def forward(self, x: Tensor) -> Tensor:
        if self.head_type == "linear" or self.head_type.startswith("mlp"):
            x = torch.flatten(x, 1)
        x = self.head(x)
        return x


@dataclasses.dataclass
class LayerForwardResult:
    """
    Forward results of each trainable block.

    output: Output just before the projection head. This is used for the next layer.
    loss: Calculateed loss value. It is calculated with the output of projection head.
    """

    output: Tensor
    loss: Tensor


class TrainableBlockConfig:
    """
    Configuration for trainable blocks.
    Args:
        cfg: Hydra configuration dictionary.
    """

    def __init__(self, cfg: dict, out_features: int = None, out_channels: int = None, input_size: int = None):
        self.out_featuers: int = out_features
        self.out_channels: int = out_channels
        self.input_size: int = input_size

        self.num_classes: int = cfg["num_classes"]
        self.head_type: str = cfg["head_type"]
        self.use_normalization: bool = True
        self.use_activation: bool = True
        self.dropout_rate: float = cfg["dropout_rate"] if "dropout_rate" in cfg else 0.2

        self.dim_in_decoder: int = 2048

        if not self.head_type in ["linear", "mlp", "conv", "conv_2", "identity"]:
            raise ValueError(f"Invalid head type: {self.head_type}")
        if out_features is None and (out_channels is None or input_size is None):
            raise ValueError("`out_features` or (`out_channels` and `input_size`) must be specified.")


class TrainableBlock1d(nn.Module):
    """
    Trainable block for 1D data.

    Args:
        block: Wrapped block.
        block_cfg: Configuration for the block.
        loss_cfg: Configuration for loss functions used in the block.
    """

    def __init__(
        self,
        block: nn.Module,
        block_cfg: dict,
        loss_cfg: dict,
    ):
        super().__init__()
        self.block: nn.Module = block
        self.block_cfg: TrainableBlockConfig = block_cfg
        self.loss_cfg: LayerWiseLossConfig = loss_cfg

        if self.block_cfg.head_type.startswith("conv"):
            raise ValueError("Convolutional head is not supported in TrainableBlock1d.")

        if self.block_cfg.use_normalization:
            self.normalization: nn.Module = nn.BatchNorm1d(self.block_cfg.out_features)
        if self.block_cfg.use_activation:
            self.activation: nn.Module = nn.ReLU()
        if self.block_cfg.dropout_rate > 0:
            self.dropout: nn.Module = nn.Dropout(p=self.block_cfg.dropout_rate)

        if self.loss_cfg.loss_type == "cross_entropy":
            if self.block_cfg.head_type == "identity" and self.block_cfg.out_featuers != self.block_cfg.num_classes:
                raise ValueError("Output dimension must be the same as the number of classes.")
            self.head = ProjectionHead(
                in_features=self.block_cfg.out_featuers,
                out_features=self.block_cfg.num_classes,
                head_type=self.block_cfg.head_type,
            )
        elif self.loss_cfg.loss_type in ["similarity", "supervised_contrastive"]:
            self.head = ProjectionHead(
                in_features=self.block_cfg.out_featuers,
                in_features=self.block_cfg.out_featuers,
                head_type=self.block_cfg.head_type,
            )
        else:
            raise ValueError("Invalid loss type: {}".format(self.loss_cfg.loss_type))
        self.loss_func = LayerWiseLoss(loss_cfg=self.loss_cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        if self.block_cfg.use_normalization:
            x = self.normalization(x)
        if self.block_cfg.use_activation:
            x = self.activation(x)
        if self.block_cfg.dropout_rate > 0:
            x = self.dropout(x)
        return x

    def forward_with_loss(self, x: Tensor, y: Tensor, model_input: Tensor) -> LayerForwardResult:
        output = self.forward(x)
        loss = self.loss_func.criterion(fx=self.head(output), x=x, y=y, model_input=model_input)
        return LayerForwardResult(output=output, loss=loss)


class TrainableBlock2d(nn.Module):
    """
    Trainable block for 1D data.

    Args:
        block: Wrapped block.
        block_cfg: Configuration for the block.
        loss_cfg: Configuration for loss functions used in the block.
    """

    def __init__(
        self,
        block: nn.Module,
        block_cfg: dict,
        loss_cfg: dict,
    ):
        super().__init__()
        self.block: nn.Module = block
        self.block_cfg: TrainableBlockConfig = block_cfg
        self.loss_cfg: LayerWiseLossConfig = loss_cfg

        if self.block_cfg.use_normalization:
            self.normalization: nn.Module = nn.BatchNorm2d(self.block_cfg.out_channels)
        if self.block_cfg.use_activation:
            self.activation: nn.Module = nn.ReLU()
        if self.block_cfg.dropout_rate > 0:
            self.dropout: nn.Module = nn.Dropout(p=self.block_cfg.dropout_rate)

        if self.block_cfg.head_type in ["linear", "mlp"]:
            # Insert pooling layer before projection head in the case of lienar or mlp head.
            # This is used to suppress the number of parameters in the projection head and avoid hitting memory limit.
            avg_pool, out_features = TrainableBlock2d.get_avg_pool(
                out_channels=self.block_cfg.out_channels,
                input_size=self.block_cfg.input_size,
                dim_in_decoder=self.block_cfg.dim_in_decoder,
            )
            if self.loss_cfg.loss_type == "cross_entropy":
                self.head = ProjectionHead(
                    in_features=out_features,
                    out_features=self.block_cfg.num_classes,
                    head_type=self.block_cfg.head_type,
                )
            elif self.loss_cfg.loss_type in ["similarity", "supervised_contrastive"]:
                self.head = ProjectionHead(
                    in_features=out_features,
                    out_features=out_features,
                    head_type=self.block_cfg.head_type,
                )
            else:
                assert False, "Unreachable"
            if avg_pool is not None:
                self.head = nn.Sequential(avg_pool, self.head)
        elif self.block_cfg.head_type in ["conv", "conv_2"]:
            if self.loss_cfg.loss_type == "cross_entropy":
                raise ValueError("Convolutional head is not supported for cross-entropy loss.")
            elif self.loss_cfg.loss_type == "similarity":
                self.head = ProjectionHead(
                    in_channels=self.block_cfg.out_channels,
                    input_size=self.block_cfg.input_size,
                    out_features=self.block_cfg.out_channels,
                    head_type=self.block_cfg.head_type,
                )
            elif self.loss_cfg.loss_type == "supervised_contrastive":
                raise ValueError("Convolutional head is not supported for supervised contrastive loss.")
            else:
                assert False, "Unreachable"
        elif self.block_cfg.head_type == "identity":
            if self.loss_cfg.loss_type == "cross_entropy":
                raise ValueError("Identity head is not supported for cross-entropy loss.")
            elif self.loss_cfg.loss_type == "similarity":
                self.head = ProjectionHead(head_type=self.block_cfg.head_type)
            elif self.loss_cfg.loss_type == "supervised_contrastive":
                # Insert average pooling layer to suppress the number of dimenstions to compute supervised contrastive loss.
                avg_pool, out_features = TrainableBlock2d.get_avg_pool(
                    out_channels=self.block_cfg.out_channels,
                    input_size=self.block_cfg.input_size,
                    dim_in_decoder=self.block_cfg.dim_in_decoder,
                )
                self.head = ProjectionHead(head_type=self.block_cfg.head_type)
                if avg_pool is not None:
                    self.head = nn.Sequential(avg_pool, self.head)
            else:
                assert False, "Unreachable"
        else:
            assert False, "Unreachable"

        self.loss_func = LayerWiseLoss(loss_cfg=self.loss_cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        if self.block_cfg.use_normalization:
            x = self.normalization(x)
        if self.block_cfg.use_activation:
            x = self.activation(x)
        if self.block_cfg.dropout_rate > 0:
            x = self.dropout(x)
        return x

    def forward_with_loss(self, x: Tensor, y: Tensor, model_input: Tensor) -> LayerForwardResult:
        output = self.forward(x)
        loss = self.loss_func.criterion(fx=self.head(output), x=x, y=y, model_input=model_input)
        return LayerForwardResult(output=output, loss=loss)

    @staticmethod
    def get_avg_pool(out_channels: int, input_size: int, dim_in_decoder: int) -> tuple[nn.Module, int]:
        """
        Get pooling layer to reduce the number of input dimensions for the projection head.
        Specifically, the input dimension is changed from `out_channels` * `input_size` * `input_size` to `dim_in_decoder` (default: 2048).
        (see https://github.com/anokland/local-loss/blob/master/train.py#L682)

        Returns:
            avg_pool: `nn.Module` class for average pooling.
            out_features: Output feature dimension of the pooling layer.
        """
        kernel_size_h, kernel_size_w = 1, 1
        size_h, size_w = input_size, input_size
        out_features = out_channels * size_h * size_w
        while out_features > dim_in_decoder and kernel_size_h < input_size:
            kernel_size_h *= 2
            size_h = ceil(input_size / kernel_size_h)
            out_features = out_channels * size_h * size_w
            if out_features > dim_in_decoder:
                kernel_size_w *= 2
                size_w = ceil(input_size / kernel_size_w)
                out_features = out_channels * size_h * size_w
        if kernel_size_h > 1 or kernel_size_w > 1:
            padding_h = (kernel_size_h * (size_h - input_size // kernel_size_h)) // 2
            padding_w = (kernel_size_w * (size_w - input_size // kernel_size_w)) // 2
            avg_pool = nn.AvgPool2d((kernel_size_h, kernel_size_w), padding=(padding_h, padding_w))
        else:
            avg_pool = None
        return avg_pool, out_features
