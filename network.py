import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size
    def forward(self, x):
        return x.view(-1, self.size) 

class BottleneckResidual(nn.Module):
    """
    Bottleneck Residual block as described in section 3.3
    in the original paper.
    """
    def __init__(self, in_dim, out_dim, s, t):
        """
        Bottleneck residual block transforming from in_dim channels to
        out_dim channels, with stride s, and expansion factor t
        """
        super(BottleneckResidual, self).__init__()

        assert s in (1, 2), "Stride for Residual Block is either 1 or 2"
        hid_dim = round(in_dim * t)
        self.stride = s
        self.match = (in_dim == out_dim)
        self.block = nn.Sequential(

            # Pointwise convolution
            nn.Conv2d(in_dim, hid_dim, 1, bias=False),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU6(inplace=True),

            # Depthwise convolution
            nn.Conv2d(hid_dim, hid_dim, 3, s, 1, bias=False),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU6(inplace=True),

            # Linear pointwise convolution
            nn.Conv2d(hid_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim))

        # Match the number of channels between input and output
        if not self.match:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim))

    def forward(self, x):
        out = self.block(x)
        if self.stride is 1:
            if not self.match:
                x = self.shortcut(x)
            return x + out
        else:
            return out


class MobileNetv2(nn.Module):
    def __init__(self, config):
        super(MobileNetv2, self).__init__()

        self.width_mult = config.width_mult
        self.classes = config.classes
        self.feature_size = config.input_size // 32
        self.features = []

        in_dim = 3
        for (operator, t, c, n, s) in config.layers:
            out_dim = round(c * self.width_mult)
            if operator == 'conv2d':
                self.features.append(nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, 3, s, 1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU6(inplace=True)))
            elif operator == 'conv2d 1x1':
                self.features.append(nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, 1, 1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU6(inplace=True)))
            elif operator == 'bottleneck':
                for i in range(n):
                    s = s if i == 0 else 1
                    self.features.append(
                        BottleneckResidual(in_dim, out_dim, s, t))
                    in_dim = out_dim
            in_dim = out_dim

        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(
            nn.AvgPool2d(self.feature_size),
            nn.Dropout(0.2, inplace=True),
            View(in_dim),
            nn.Conv2d(in_dim, self.classes, 1, 1, bias=False))
        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        out = self.classifier(x)
        out = out.view(-1, self.classes)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)