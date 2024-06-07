import dataclasses

import torch
from torch import nn, Tensor

from models.trainable_block import TrainableBlock1d, TrainableBlock2d, LayerForwardResult
from models.layer_wise_model_spec import LayerWiseModelSpec


@dataclasses.dataclass
class ModelForwardResult:
    """
    Forward result of the model.

    feature: Output just before the classification head.
    layer_wise_loss_list: List of layer-wise losses.
    classification_loss: loss of the final classification head.
    """

    feature: Tensor
    layer_wise_loss_list: list[Tensor]
    classification_loss: Tensor


@dataclasses.dataclass
class ModelForwardResultSpecifiedLayer:
    """
    Forward result of the model for sequential layer-wise training.

    feature: Output just after the specified layer. If the final classification head is specified, it is the output
        just before the classification head.
    loss: Loss of the specified layer.
    layer_index: Index of the specified layer.
    """

    feature: Tensor
    loss: Tensor
    layer_index: int


class LayerWiseModel(nn.Module):
    """
    Base class for layer-wise models.
    """

    def __init__(self, model_spec: LayerWiseModelSpec, num_classes: int):
        super().__init__()
        self.layers: nn.ModuleList = model_spec.layers
        # Classifier at the end of the model, like E2E training.
        self.classifier: nn.Module = nn.Linear(in_features=model_spec.out_features, out_features=num_classes)
        self.criterion: nn.Module = nn.CrossEntropyLoss()

        # Number of trainable layers
        self._num_trainable_layers: int = 1
        for layer in self.layers:
            if isinstance(layer, TrainableBlock1d) or isinstance(layer, TrainableBlock2d):
                self._num_trainable_layers += 1

    def forward(self, x: Tensor) -> Tensor:
        """
        Fowward input `x` through model.
        Returns:
            Output after the final classifier.
        """
        for layer in self.layers:
            x = layer(x)
        assert x.dim() == 2
        return self.classifier(x)

    def forward_with_loss(self, x: Tensor, y: Tensor) -> ModelForwardResult:
        """
        Forward input `x` through model and compute layer-wise losses.

        Args:
            x: Input data.
            y: Target label.
        Returns:
            `ModelForwardResult` object.
        """
        layer_wise_loss_list = []
        model_input = x.clone()
        for layer in self.layers:
            if isinstance(layer, TrainableBlock1d) or isinstance(layer, TrainableBlock2d):
                layer_forward_result: LayerForwardResult = layer.forward_with_loss(x, y, model_input)

                layer_wise_loss_list.append(layer_forward_result.loss)
                x = layer_forward_result.output.detach()
            else:
                x = layer(x).detach()
        classification_loss: Tensor = self.criterion(self.classifier(x), y)

        return ModelForwardResult(
            feature=x, layer_wise_loss_list=layer_wise_loss_list, classification_loss=classification_loss
        )

    def forward_with_loss_aug(self, x1: Tensor, x2: Tensor, y: Tensor) -> ModelForwardResult:
        """
        Forward augmented inputs `x1` and `x2`, mainly in the case of supervised contrastive learning.

        Args:
            x1: Input data.
            x2: Input data.
            y: Target label.
        Returns:
            `ModelForwardResult` object.
        """
        x = torch.cat([x1, x2], dim=0)
        batch_size = y.size(0)
        layer_wise_loss_list = []
        model_input = x.clone()
        for layer in self.layers:
            if isinstance(layer, TrainableBlock1d) or isinstance(layer, TrainableBlock2d):
                layer_forward_result: LayerForwardResult = layer.forward_with_loss(x, y, model_input)
                layer_wise_loss_list.append(layer_forward_result.loss)
                x = layer_forward_result.output.detach()
            else:
                x = layer(x).detach()
        x1, _ = torch.split(x, [batch_size, batch_size], dim=0)
        classification_loss: Tensor = self.criterion(self.classifier(x1), y)

        return ModelForwardResult(
            feature=x, layer_wise_loss_list=layer_wise_loss_list, classification_loss=classification_loss
        )

    def forward_with_loss_sequentially(
        self, x: Tensor, y: Tensor, layer_index: int = 0
    ) -> ModelForwardResultSpecifiedLayer:
        """
        Forward input `x` through the model until the layer specified by `layer_index`.
        Also return the loss at the specified layer.
        It is used for the sequential layer-wise training. (see section B.2 in the paper)

        Args:
            x: Input data.
            y: Target label.
            layer_index: Layer index to be trained. It takes the value from 0 to `self.num_trainable_layers` - 1.
        Returns:
            `ModelForwardResultSpecifiedLayer` object containing information for the specified layer.
        """
        assert 0 <= layer_index < self.num_trainable_layers

        model_input = x.clone()
        loss = None
        num_layers = 0
        for layer in self.layers:
            if isinstance(layer, TrainableBlock1d) or isinstance(layer, TrainableBlock2d):
                if layer_index == num_layers:
                    layer_forward_result: LayerForwardResult = layer.forward_with_loss(x, y, model_input)
                    loss = layer_forward_result.loss
                    x = layer_forward_result.output.detach()
                else:
                    x = layer(x).detach()
                num_layers += 1
            else:
                x = layer(x).detach()
            if num_layers > layer_index:
                break

        if layer_index == self.num_trainable_layers - 1:
            assert layer_index == num_layers
            loss = self.criterion(self.classifier(x), y)

        assert loss is not None
        return ModelForwardResultSpecifiedLayer(feature=x, loss=loss, layer_index=layer_index)

    @property
    def encoder(self) -> nn.Module:
        layers = self.layers
        # Remove the final classifier of VGG.
        # This is a temporary solution.
        while not isinstance(layers[-1], nn.Flatten):
            layers = layers[:-1]
        return nn.Sequential(*layers)

    @property
    def num_trainable_layers(self) -> int:
        return self._num_trainable_layers
