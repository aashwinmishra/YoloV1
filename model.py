import torch
import torch.nn as nn
from typing import List


architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1,),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]  # (kernel_size, filters, stride, padding), "M"=maxpool(2), List[-1]: repeats


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, xb):
        return self.leakyrelu(self.batchnorm(self.conv(xb)))


class YOLOv1(nn.Module):
    def __init__(self, in_channels: int = 3, **kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture: List):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layer = CNNBlock(in_channels=in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3])
                layers.append(layer)
                in_channels = x[1]
            elif type(x) == str:
                layer = nn.MaxPool2d(2, 2)
                layers.append(layer)
            elif type(x) == list:
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layer = CNNBlock(in_channels=in_channels, out_channels=x[0][1], kernel_size=x[0][0],
                                     stride=x[0][2],
                                     padding=x[0][3])
                    layers.append(layer)
                    layer = CNNBlock(in_channels=x[0][1], out_channels=x[1][1], kernel_size=x[1][0],
                                     stride=x[1][2],
                                     padding=x[1][3])
                    layers.append(layer)
                    in_channels = x[1][1]
        return nn.Sequential(*layers)

    @staticmethod
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5))
        )


def test(split_size=7, num_boxes=2, num_classes=20):
    model = YOLOv1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)




