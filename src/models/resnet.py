import torch
from torch import nn, Tensor
import torch.nn.functional as F

# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Model(nn.Module):
    """
    ResNet model without the final fully connected layer.
    """

    def __init__(self, block, num_blocks, in_channel: int = 3):
        super().__init__()
        self.init_in_planes = 64
        self.in_planes = self.init_in_planes

        self.conv1 = nn.Conv2d(in_channel, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.init_in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.init_in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.init_in_planes * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.init_in_planes * 8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def get_model_settings(model_name: str):
    if model_name == "resnet18":
        block = BasicBlock
        dim_out = 512
        num_blocks = [2, 2, 2, 2]
    elif model_name == "resnet34":
        block = BasicBlock
        dim_out = 512
        num_blocks = [3, 4, 6, 3]
    elif model_name == "resnet50":
        block = Bottleneck
        dim_out = 2048
        num_blocks = [3, 4, 6, 3]
    elif model_name == "resnet101":
        block = Bottleneck
        dim_out = 2048
        num_blocks = [3, 4, 23, 3]
    elif model_name == "resnet152":
        block = Bottleneck
        dim_out = 2048
        num_blocks = [3, 8, 36, 3]
    else:
        raise ValueError("ResNet model name {} is invalid".format(model_name))
    return block, dim_out, num_blocks


class SupConResNet(nn.Module):
    """
    ResNet encoder + projection head.
    This model is used for supervised contrastive learning.
    """

    def __init__(self, model_name: str = "resnet50", head_type: str = "mlp", feat_dim: int = 128):
        super().__init__()
        block, dim_out, num_blocks = get_model_settings(model_name)
        self.encoder = Model(block, num_blocks)
        if head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(dim_out, dim_out),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out, feat_dim),
            )
        elif head_type == "linear":
            self.head = nn.Linear(dim_out, feat_dim)
        else:
            raise ValueError("Invalid head type {}".format(head_type))

    def forward(self, x: Tensor) -> Tensor:
        feature = self.encoder(x)
        return F.normalize(self.head(feature), dim=1)


class SupCEResNet(nn.Module):
    """
    ResNet encoder + classifier.
    This model is used for supervised classification.
    """

    def __init__(self, model_name: str = "resnet50", num_classes: int = 10):
        super().__init__()
        block, dim_out, num_blocks = get_model_settings(model_name)
        self.encoder = Model(block, num_blocks)
        self.fc = nn.Linear(dim_out, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.encoder(x))


# check
if __name__ == "__main__":
    net = SupConResNet("resnet18")
    y = net(torch.randn(64, 3, 32, 32))
    assert y.shape == (64, 128)

    net = SupConResNet("resnet50")
    y = net(torch.randn(64, 3, 32, 32))
    assert y.shape == (64, 128)

    net = SupCEResNet("resnet18")
    y = net(torch.randn(64, 3, 32, 32))
    assert y.shape == (64, 10)

    net = SupCEResNet("resnet50")
    y = net(torch.randn(64, 3, 32, 32))
    assert y.shape == (64, 10)
