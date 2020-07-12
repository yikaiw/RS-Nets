"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math
from .parallel import ModuleParallel, BatchNorm2dParallel

__all__ = ['parallel_mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride, num_parallel):
    return nn.Sequential(
        ModuleParallel(nn.Conv2d(inp, oup, 3, stride, 1, bias=False)),
        BatchNorm2dParallel(oup, num_parallel),
        ModuleParallel(nn.ReLU6(inplace=True))
    )


def conv_1x1_bn(inp, oup, num_parallel):
    return nn.Sequential(
        ModuleParallel(nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
        BatchNorm2dParallel(oup, num_parallel),
        ModuleParallel(nn.ReLU6(inplace=True))
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, num_parallel):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                ModuleParallel(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
                BatchNorm2dParallel(hidden_dim, num_parallel),
                ModuleParallel(nn.ReLU6(inplace=True)),
                # pw-linear
                ModuleParallel(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                BatchNorm2dParallel(oup, num_parallel),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                ModuleParallel(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)),
                BatchNorm2dParallel(hidden_dim, num_parallel),
                ModuleParallel(nn.ReLU6(inplace=True)),
                # dw
                ModuleParallel(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
                BatchNorm2dParallel(hidden_dim, num_parallel),
                ModuleParallel(nn.ReLU6(inplace=True)),
                # pw-linear
                ModuleParallel(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                BatchNorm2dParallel(oup, num_parallel),
            )
        self.num_parallel = num_parallel

    def forward(self, x):
        if self.identity:
            out = self.conv(x)
            return [x[i] + out[i] for i in range(self.num_parallel)]
        else:
            return self.conv(x)


class ParallelMobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., num_parallel=5):
        super(ParallelMobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2, num_parallel)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, num_parallel))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel, num_parallel)
        self.avgpool = ModuleParallel(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = ModuleParallel(nn.Linear(output_channel, num_classes))

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = [t.view(t.size(0), -1) for t in x]
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def parallel_mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return ParallelMobileNetV2(**kwargs)
