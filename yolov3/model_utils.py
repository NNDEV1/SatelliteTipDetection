import torch
import torch.nn as nn

class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.bn_act = bn_act
    
    def forward(self, x):

        if self.bn_act:

            x = self.leaky(self.bn(self.conv(x)))

        else:

            x = self.conv(x)

        return x


class ResidualBlock(nn.Module):

    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()

        self.layers = nn.ModuleList()

        for repeat in range(num_repeats):

            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels//2, kernel_size=1),
                    CNNBlock(channels//2, channels, kernel_size=3, padding=1)
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):

        for layer in self.layers:

            x = layer(x) + x if self.use_residual else layer(x)

        return x
        

class ScaledPrediction(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.block = nn.Sequential(
            CNNBlock(in_channels, in_channels*2, kernel_size=3, padding=1),
            CNNBlock(in_channels*2, (num_classes + 5) * 3, bn_act=False, kernel_size=1)
        )

        self.num_classes = num_classes

    def forward(self, x):

        x = self.block(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)

        return x
