import abc
import dataclasses
from typing import cast

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from omegaconf import DictConfig

from src.models import resnet
from src.models.resnet import BasicBlock, Bottleneck
from src.util import WholeAvgPool2d


def calc_signal_propagation_loss(
    f_data: Tensor, signal: Tensor, f_signal: Tensor, loss_type: str
) -> Tensor:
    """
    Compute loss function based on "Signal Propagation" paper (https://arxiv.org/abs/2204.01723).
    Args:
        f_data: Data output of the block.
        signal: Signal input, i.e., propagated class information.
        f_signal: Signal output of the block.
        loss_type: Style of target similarity calculation.
            "hard": Choose top k similar samples and use 0-1 similarity. (k = 6 in the original paper)
            "soft": Use softmax similarity.
    """
    # Normalize dot product inspired by Attention
    target_dim = signal.numel() / signal.size(0)

    # Bring (f_data)_i^T (f_signal)_j closer to (signal)_i^T (signal)_j
    target_similarity = (
        signal.view(signal.size(0), -1)
        @ signal.view(signal.size(0), -1).T
        / (target_dim**0.5)
    )

    if loss_type == "hard":
        target_similarity_tmp = target_similarity.topk(6, dim=1)[1]
        target_similarity = (
            F.one_hot(target_similarity_tmp, target_similarity.shape[1])
            .sum(1)
            .clamp(0, 1)
            .float()
        )
    elif loss_type == "soft":
        target_similarity = F.softmax(target_similarity, dim=-1)
    else:
        raise ValueError("Unknown loss_type: {}".format(loss_type))

    output_dim = f_signal.numel() / f_signal.size(0)
    output_similarity = (
        f_data.view(f_data.size(0), -1)
        @ f_signal.view(f_signal.size(0), -1).T
        / (output_dim**0.5)
    )
    loss = torch.sum(
        -target_similarity * F.log_softmax(output_similarity, dim=-1), dim=-1
    ).mean()

    return loss


class SPTrainableBlockConfig:
    """
    Configuration for trainable blocks.
    Args:
        cfg: Hydra configuration dictionary.
    """

    def __init__(
        self,
        out_features: int | None = None,
        out_channels: int | None = None,
        use_activation: bool = True,
        normalize_method: str | None = None,
        dropout_rate: float = 0.0,
    ):
        self.out_features: int | None = out_features
        self.out_channels: int | None = out_channels
        self.use_activation: bool = use_activation
        self.normalize_method: str | None = normalize_method
        self.dropout_rate: float = dropout_rate


class SPTrainableBlock1d(nn.Module):
    """
    Trainable 1d block with signal propagation loss.

    Args:
        block: Wrapped block.
        block_cfg: Configuration for the block.
    """

    def __init__(
        self,
        block: nn.Module,
        block_cfg: SPTrainableBlockConfig,
    ):
        super().__init__()
        self.block: nn.Module = block
        self.block_cfg: SPTrainableBlockConfig = block_cfg

        if block_cfg.use_activation:
            self.activation: nn.Module = nn.LeakyReLU()
        if block_cfg.dropout_rate > 0:
            self.dropout: nn.Module = nn.Dropout(p=self.block_cfg.dropout_rate)

        if block_cfg.out_features is None:
            if isinstance(block, nn.Linear):
                block_cfg.out_features = block.out_features
            else:
                raise ValueError(
                    "block_cfg.out_features must be specified if block is not nn.Linear"
                )

        if block_cfg.normalize_method is not None:
            if block_cfg.normalize_method == "batch_norm":
                self.normalization: nn.Module = nn.BatchNorm1d(block_cfg.out_features)
            elif block_cfg.normalize_method == "layer_norm":
                self.normalization: nn.Module = nn.LayerNorm(block_cfg.out_features)
            else:
                raise ValueError(
                    "Normalization method {} is not supported".format(
                        block_cfg.normalize_method
                    )
                )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        if self.block_cfg.normalize_method is not None:
            x = self.normalization(x)
        if self.block_cfg.use_activation:
            x = self.activation(x)
        if self.block_cfg.dropout_rate > 0:
            x = self.dropout(x)
        return x

    def forward_signal(self, signal: Tensor) -> Tensor:
        return self.forward(signal)


class SPTrainableBlock2d(nn.Module):
    """
    Trainable 2d block with signal propagation loss.

    Args:
        block: Wrapped block.
        block_cfg: Configuration for the block.
    """

    def __init__(
        self,
        block: nn.Module,
        block_cfg: SPTrainableBlockConfig,
    ):
        super().__init__()
        self.block: nn.Module = block
        self.block_cfg: SPTrainableBlockConfig = block_cfg

        if block_cfg.use_activation:
            self.activation: nn.Module = nn.LeakyReLU()
        if block_cfg.dropout_rate > 0:
            self.dropout: nn.Module = nn.Dropout(p=self.block_cfg.dropout_rate)

        if block_cfg.out_channels is None:
            if isinstance(block, nn.Conv2d):
                block_cfg.out_channels = block.out_channels
            else:
                raise ValueError(
                    "block_cfg.channels must be specified if block is not nn.Conv2d"
                )

        if block_cfg.normalize_method is not None:
            if block_cfg.normalize_method == "batch_norm":
                self.normalization: nn.Module = nn.BatchNorm2d(block_cfg.out_channels)
            elif block_cfg.normalize_method == "instance_norm":
                self.normalization: nn.Module = nn.InstanceNorm2d(
                    block_cfg.out_channels, affine=True
                )
            else:
                raise ValueError(
                    "Normalization method {} is not supported".format(
                        block_cfg.normalize_method
                    )
                )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        if self.block_cfg.normalize_method is not None:
            x = self.normalization(x)
        if self.block_cfg.use_activation:
            x = self.activation(x)
        if self.block_cfg.dropout_rate > 0:
            x = self.dropout(x)
        return x

    def forward_signal(self, signal: Tensor) -> Tensor:
        return self.forward(signal)


class SPInputBlock(nn.Module):
    """
    Block to convert the input data and signal to the common input space.

    Args:
        input_size: Width (height) of the data after this block. This equals to the width (height) of the input data.
        input_channels: Number of channels of the data after this block.
    """

    def __init__(self, cfg: DictConfig, input_size: int, input_channels: int):
        super().__init__()
        self.num_classes: int = cfg["dataset"]["num_classes"]
        self.input_size: int = input_size
        self.input_channels: int = input_channels

        self.input_block_data = nn.Sequential(
            nn.Conv2d(
                3, input_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
        )
        self.input_block_signal = nn.Sequential(
            nn.Linear(
                self.num_classes, input_channels * input_size * input_size, bias=False
            ),
            nn.LayerNorm(input_channels * input_size * input_size),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.input_block_data(x)

    def forward_signal(self, signal: Tensor) -> Tensor:
        if signal.size(1) != self.num_classes:
            raise ValueError("`signal` should be one-hot encoded.")
        return self.input_block_signal(signal).view(
            -1, self.input_channels, self.input_size, self.input_size
        )


@dataclasses.dataclass
class SPForwardResult:
    data: Tensor
    signal: Tensor
    layer_wise_loss_list: list[Tensor]
    classification_loss: Tensor


class SPLayerWiseModel(nn.Module, abc.ABC):
    """
    Base class for layer-wise models.
    Subclasses of this class have to pass `layers` to the constructor of this class.
    They also have to implement `forward` method.
    """

    def __init__(
        self,
        cfg: DictConfig,
        layers: nn.ModuleList,
        input_channels: int,
        output_features: int,
    ):
        super().__init__()
        self.layers: nn.ModuleList = layers
        self.input_channels: int = input_channels
        self.output_features: int = output_features
        self.loss_type: str = cfg["loss_type"]
        self.input_size: int = cfg["dataset"]["size"]
        self.num_classes: int = cfg["dataset"]["num_classes"]

        if self.loss_type not in ["hard", "soft"]:
            raise ValueError("Loss type {} is not supported.".format(self.loss_type))

        self.input_block = SPInputBlock(
            cfg=cfg, input_size=self.input_size, input_channels=input_channels
        )
        self.linear_head = nn.Linear(self.output_features, self.num_classes)

    def forward(self, data: Tensor) -> Tensor:
        """
        Fowward input data through model without signal.
        Returns:
            Output after the classification head.
        """
        for layer in [self.input_block, *self.layers]:
            data = layer(data)
        return self.forward_head(data)

    def forward_with_signal(self, data: Tensor, label: Tensor) -> SPForwardResult:
        """ "
        Forward input data and label signal through the model and calculate the loss.
        Args:
            data: Input data. Shape is (batch_size, 3, input_size, input_size).
            label: Label information. Shape is (batch_size).
        """
        # One-hot encoding
        signal = label.detach().clone()
        signal = F.one_hot(signal, num_classes=self.num_classes).float()

        data, signal, layer_wise_loss_list = self.forward_body(data, signal)
        output = self.forward_head(data)
        classification_loss = F.cross_entropy(output, label)

        return SPForwardResult(
            data=data,
            signal=signal,
            layer_wise_loss_list=layer_wise_loss_list,
            classification_loss=classification_loss,
        )

    def forward_body(
        self, data: Tensor, signal: Tensor
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """
        Forward input data and label signal through self.input_block and self.layers.
        """
        loss_list = []
        for layer in [self.input_block, *self.layers]:
            if (
                isinstance(layer, SPInputBlock)
                or isinstance(layer, SPTrainableBlock1d)
                or isinstance(layer, SPTrainableBlock2d)
            ):
                f_data, f_signal = layer(data), layer.forward_signal(signal)
                loss = calc_signal_propagation_loss(
                    f_data=f_data,
                    signal=signal,
                    f_signal=f_signal,
                    loss_type=self.loss_type,
                )
                loss_list.append(loss)
                data, signal = (
                    layer(data).detach(),
                    layer.forward_signal(signal).detach(),
                )
            else:
                data, signal = layer(data).detach(), layer(signal).detach()

        return data, signal, loss_list

    def forward_head(self, data: Tensor) -> Tensor:
        """
        Auxiliary head for classification.
        """
        if data.dim() == 4:
            data = data.view(data.size(0), -1)
        assert data.dim() == 2

        return self.linear_head(data)


class SPLayerWiseCNN(SPLayerWiseModel):
    """
    Toy CNN model for signal propagation training.
    Args:
        cfg: Hydra configuration dictionary.
    """

    def __init__(self, cfg: DictConfig):
        normalize_method: str = "instance_norm"
        dropout_rate: float = 0.2

        layers = nn.ModuleList(
            [
                SPTrainableBlock2d(
                    block=nn.Conv2d(
                        in_channels=32, out_channels=32, kernel_size=(3, 3)
                    ),
                    block_cfg=SPTrainableBlockConfig(
                        normalize_method=normalize_method,
                        dropout_rate=dropout_rate,
                    ),
                ),
                SPTrainableBlock2d(
                    block=nn.Conv2d(
                        in_channels=32, out_channels=64, kernel_size=(3, 3)
                    ),
                    block_cfg=SPTrainableBlockConfig(
                        normalize_method=normalize_method,
                        dropout_rate=dropout_rate,
                    ),
                ),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Flatten(),
                SPTrainableBlock1d(
                    block=nn.Linear(
                        in_features=(cfg["dataset"]["size"] - 4) ** 2 * 64 // 4,
                        out_features=128,
                    ),
                    block_cfg=SPTrainableBlockConfig(
                        normalize_method="layer_norm",
                        dropout_rate=dropout_rate,
                    ),
                ),
            ]
        )
        super().__init__(
            cfg=cfg,
            layers=layers,
            input_channels=32,
            output_features=128,
        )


class SPLayerWiseResNet(SPLayerWiseModel):
    """
    ResNet model for signal propagation training.
    Args:
        cfg: Hydra configuration dictionary.
        model_name: name of the ResNet model. Currently, `resnet18`, `resnet34`, `resnet50`, `resnet101`, and
                `resnet152` are supported.
    """

    def __init__(
        self,
        cfg: DictConfig,
        model_name: str,
    ):
        self.cfg = cfg
        self.normalize_method: str = "instance_norm"
        self.dropout_rate: float = 0.2

        block, dim_out, num_blocks = resnet.get_model_settings(model_name)

        self.init_in_planes = 64
        self.in_planes = self.init_in_planes

        layers = nn.ModuleList()
        layers.extend(
            self._make_layer(block, self.init_in_planes, num_blocks[0], stride=1)
        )
        layers.extend(
            self._make_layer(block, self.init_in_planes * 2, num_blocks[1], stride=2)
        )
        layers.extend(
            self._make_layer(block, self.init_in_planes * 4, num_blocks[2], stride=2)
        )
        layers.extend(
            self._make_layer(block, self.init_in_planes * 8, num_blocks[3], stride=2)
        )
        layers.append(WholeAvgPool2d())
        layers.append(nn.Flatten())
        super().__init__(
            cfg=cfg,
            layers=layers,
            input_channels=self.init_in_planes,
            output_features=dim_out,
        )

    def _make_layer(
        self,
        block: type[BasicBlock | Bottleneck],
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.ModuleList:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        for stride in strides:
            layers.append(
                SPTrainableBlock2d(
                    block=block(in_planes=self.in_planes, planes=planes, stride=stride),
                    block_cfg=SPTrainableBlockConfig(
                        out_channels=planes * block.expansion,
                        normalize_method=self.normalize_method,
                        dropout_rate=self.dropout_rate,
                    ),
                )
            )
            self.in_planes = planes * block.expansion
        return layers


class SPLayerWiseVGG(SPLayerWiseModel):
    """
    VGG model for signal propagation training.
    Args:
        cfg: Hydra configuration dictionary.
        model_name: name of VGG models. Currently, `vgg11`, `vgg13`, `vgg16`, `vgg19` are supported.
    """

    # fmt: off
    cfgs: dict[str, list[str | int]] = {
        "vgg6b": [128, "M", 256, "M", 512, "M", 512, "M"],
        "vgg8b": [128, 256, "M", 256, 512, "M", 512, "M", 512, "M"],
        "vgg11b": [128, 128, 128, 256, "M", 256, 512, "M", 512, 512, "M", 512, "M"],
        "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    }
    # fmt: on

    def __init__(
        self,
        cfg: DictConfig,
        model_name: str,
    ):
        self.cfg = cfg
        self.normalize_method: str = "instance_norm"
        self.dropout_rate: float = 0.2

        if model_name not in self.cfgs:
            raise ValueError("Invalid model name: {}".format(model_name))
        model_cfg = self.cfgs[model_name]
        input_channels = model_cfg[0]
        assert isinstance(input_channels, int)

        layers, output_size = self._make_layers(
            model_cfg=model_cfg,
            input_size=cfg["dataset"]["size"],
            input_channels=input_channels,
        )

        out_features: int = 1024
        layers.append(nn.Flatten())
        layers.append(
            SPTrainableBlock1d(
                block=nn.Linear(
                    in_features=512 * output_size * output_size,
                    out_features=out_features,
                ),
                block_cfg=SPTrainableBlockConfig(
                    normalize_method="layer_norm",
                    dropout_rate=self.dropout_rate,
                ),
            )
        )
        super().__init__(
            cfg=cfg,
            layers=layers,
            input_channels=input_channels,
            output_features=out_features,
        )

    def _make_layers(
        self, model_cfg: list[str | int], input_size: int, input_channels: int
    ) -> tuple[nn.ModuleList, int]:
        layers: nn.ModuleList = nn.ModuleList()
        for v in model_cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                input_size = input_size // 2
            else:
                v = cast(int, v)
                layers.append(
                    SPTrainableBlock2d(
                        block=nn.Conv2d(
                            in_channels=input_channels,
                            out_channels=v,
                            kernel_size=3,
                            padding=1,
                        ),
                        block_cfg=SPTrainableBlockConfig(
                            out_channels=v,
                            normalize_method=self.normalize_method,
                            dropout_rate=self.dropout_rate,
                        ),
                    )
                )
                input_channels = v
        return layers, input_size
