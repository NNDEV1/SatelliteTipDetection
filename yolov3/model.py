import torch
import torch.nn as nn
from model_utils import ScalePrediction, CNNBlock, ResidualBlock

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class YOLOv3(nn.Module):

    def __init__(self, num_classes=1, in_channels=3):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self.create_conv_layers()

    def forward(self, x):

        outputs = []
        route_connections = []

        for layer in self.layers:
            
            if isinstance(layer, ScaledPrediction):

                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:

                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):

                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def create_conv_layers(self):

        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:

            if isinstance(module, tuple):

                out_channels, kernel_size, stride = module

                layers.append(
                    CNNBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1 if kernel_size ==3 else 0)
                )

                in_channels = out_channels

            elif isinstance(module, list):

                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))

            elif isinstance(module, str):

                if module == "S":

                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels//2, kernel_size=1),
                        ScaledPrediction(in_channels//2, num_classes=self.num_classes)
                    ]

                    in_channels = in_channels // 2

                elif module == "U":

                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3

        return layers
