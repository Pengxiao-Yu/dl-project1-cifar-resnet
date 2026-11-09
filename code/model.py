"""CIFAR residual networks, initialized from scratch (no downloaded weights).
Basic-block design follows He et al., CVPR 2016, https://arxiv.org/abs/1512.03385.
Implementation authored for this project; CIFAR stem uses stride 1 without maxpool.
"""

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = (
            nn.Identity()
            if cin == cout and stride == 1
            else nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(y)) + self.shortcut(x))


class CifarResNet(nn.Module):
    def __init__(self, width=40, blocks=(2, 2, 2, 2), classes=10):
        super().__init__()
        self.width = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        layers = []
        cin = width
        for stage, n in enumerate(blocks):
            cout = width * 2**stage
            for i in range(n):
                layers.append(BasicBlock(cin, cout, 2 if stage > 0 and i == 0 else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        assert (
            parameter_count(self) <= 5_000_000
        ), "Assignment parameter budget exceeded"

    def forward(self, x):
        return self.fc(self.pool(self.layers(self.stem(x))).flatten(1))


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
