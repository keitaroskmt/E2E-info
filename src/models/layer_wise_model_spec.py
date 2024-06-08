import abc
from typing import cast

from torch import nn

from models.trainable_block import TrainableBlock1d, TrainableBlock2d, TrainableBlockConfig
from models.layer_wise_loss import LayerWiseLossConfig
from models.resnet import get_model_settings


class LayerWiseModelSpec(abc.ABC):
    """
    Base class for layer-wise model specification.
    """

    @property
    @abc.abstractmethod
    def layers(self) -> nn.ModuleList:
        """
        List of layers.
        """
        pass

    @property
    @abc.abstractmethod
    def out_features(self) -> int:
        """
        Number of output features.
        """
        pass


class LayerWiseResNetSpec(LayerWiseModelSpec):
    """
    ResNet model for layer-wise training.

    Args:
        cfg: Hydra configuration dictionary.
    """

    def __init__(self, cfg: dict):
        self.input_size = cfg["dataset"]["size"]

        block, dim_out, num_blocks = get_model_settings(cfg["model"]["name"])
        self._out_features: int = dim_out

        self.init_in_planes = 64
        self.in_planes = self.init_in_planes
        self._layers: nn.ModuleList = nn.ModuleList(
            [
                TrainableBlock2d(
                    block=nn.Conv2d(
                        in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
                    ),
                    block_cfg=TrainableBlockConfig(cfg=cfg, out_channels=self.in_planes, input_size=self.input_size),
                    loss_cfg=LayerWiseLossConfig(cfg=cfg),
                )
            ]
        )
        self._layers.extend(self._make_layer(block, self.init_in_planes, num_blocks[0], stride=1, cfg=cfg))
        self._layers.extend(self._make_layer(block, self.init_in_planes * 2, num_blocks[1], stride=2, cfg=cfg))
        self._layers.extend(self._make_layer(block, self.init_in_planes * 4, num_blocks[2], stride=2, cfg=cfg))
        self._layers.extend(self._make_layer(block, self.init_in_planes * 8, num_blocks[3], stride=2, cfg=cfg))
        self._layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self._layers.append(nn.Flatten())

    @property
    def layers(self) -> nn.ModuleList:
        return self._layers

    @property
    def out_features(self) -> int:
        return self._out_features

    def _make_layer(self, block, planes, num_blocks, stride, cfg) -> nn.ModuleList:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: nn.ModuleList = nn.ModuleList()
        for stride in strides:
            block_instance: nn.Module = block(self.in_planes, planes, stride)
            self.in_planes = planes * block.expansion
            if stride == 2:
                self.input_size = (self.input_size + 1) // 2

            # Note that the activations are included within `BasicBlock` and `Bottleneck` of ResNet.
            block_cfg: TrainableBlockConfig = TrainableBlockConfig(
                cfg=cfg, out_channels=planes, input_size=self.input_size
            )
            block_cfg.use_normalization = False
            block_cfg.use_activation = False
            block_cfg.dropout_rate = 0.0
            layers.append(
                TrainableBlock2d(
                    block=block_instance,
                    block_cfg=block_cfg,
                    loss_cfg=LayerWiseLossConfig(cfg=cfg),
                )
            )
            self.mlp_ratio_cur += self.mlp_step
        return layers


class LayerWiseVGGSpec(LayerWiseModelSpec):
    """
    VGG model for layer-wise training.

    Args:
        cfg: Hydra configuration dictionary.
    """

    cfgs: dict[str, list[str | int]] = {
        "vgg6b": [128, "M", 256, "M", 512, "M", 512, "M"],
        "vgg8b": [128, 256, "M", 256, 512, "M", 512, "M", 512, "M"],
        "vgg11b": [128, 128, 128, 256, "M", 256, 512, "M", 512, 512, "M", 512, "M"],
        "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    }

    def __init__(self, cfg: dict):
        model_name = cfg["model"]["name"]
        input_size = cfg["dataset"]["size"]
        if model_name not in self.cfgs:
            raise ValueError("Invalid model name: {}".format(model_name))

        self._out_features: int = 512

        model_cfg = self.cfgs[model_name]
        in_channels = model_cfg[0]

        self._layers = nn.ModuleList()
        layers, output_size = self._make_layers(
            model_cfg=model_cfg,
            in_channels=in_channels,
            input_size=input_size,
            cfg=cfg,
        )
        self._layers.extend(layers)

        self._layers.append(nn.Flatten())
        block_cfg = TrainableBlockConfig(cfg=cfg, out_features=self.out_features)
        block_cfg.head_type = "linear" if cfg["head_type"].startswith("conv") else cfg["head_type"]
        self._layers.append(
            TrainableBlock1d(
                block=nn.Linear(in_features=512 * output_size * output_size, out_features=self.out_features),
                block_cfg=block_cfg,
                loss_cfg=LayerWiseLossConfig(cfg=cfg),
            )
        )
        self._layers.append(
            TrainableBlock1d(
                block=nn.Linear(in_features=self.out_features, out_features=self.out_features),
                block_cfg=block_cfg,
                loss_cfg=LayerWiseLossConfig(cfg=cfg),
            )
        )

    @property
    def layers(self) -> nn.ModuleList:
        return self._layers

    @property
    def out_features(self) -> int:
        return self._out_features

    def _make_layers(
        self,
        model_cfg: list[str | int],
        in_channels: int,
        input_size: int,
        cfg: dict,
    ) -> tuple[nn.ModuleList, int]:
        layers: nn.ModuleList = nn.ModuleList()
        for v in model_cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                input_size = input_size // 2
            else:
                v = cast(int, v)
                layers.append(
                    TrainableBlock2d(
                        block=nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, padding=1),
                        block_cfg=TrainableBlockConfig(cfg=cfg, out_channels=v, input_size=input_size),
                        loss_cfg=LayerWiseLossConfig(cfg=cfg),
                    )
                )
                in_channels = v
        return layers, input_size
