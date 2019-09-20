# BSD 2-Clause License

# Copyright (c) 2019 wangvation. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

from torch import nn
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import AvgPool2d
from torch.nn import Softmax2d
from torch.nn import Linear
from torch.nn import ReLU6
from torch.nn.functional import relu6
from torch.nn.functional import relu
from torch.nn.functional import log_softmax


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


class DepthSepConv(nn.Module):
  """docstring for Depthwise Separable Convolution"""

  def __init__(self,
               in_channels,
               out_channels,
               ksize=3,
               stride=1,
               padding=1,
               alpha=1):
    super(DepthSepConv, self).__init__()
    in_channels = _make_divisible(in_channels * alpha, 8)
    out_channels = _make_divisible(out_channels * alpha, 8)
    self.depthwise_conv = Conv2d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=ksize,
                                 stride=stride,
                                 padding=padding,
                                 groups=in_channels)

    self.bn1 = BatchNorm2d(in_channels)

    self.pointwise_conv = Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 groups=1)
    self.bn2 = BatchNorm2d(out_channels)

  def forward(self, x):
    x = self.depthwise_conv(x)
    x = self.bn1(x)
    x = relu(x)
    x = self.pointwise_conv(x)
    x = self.bn2(x)
    x = relu(x)
    return x


class MobileNetV1(nn.Module):
  """
  docstring for MobileNetV1
  MobileNetV1 Body Architecture
  | Type / Stride | Filter Shape        | Input Size     | Output Size      |
  | :------------ | :------------------ | :------------- | :-------------   |
  | Conv / s2     | 3 × 3 × 3 × 32      | 224 x 224 x 3  | 112 x 112 x 32   |
  | Conv dw / s1  | 3 × 3 × 32 dw       | 112 x 112 x 32 | 112 x 112 x 32   |
  | Conv / s1     | 1 × 1 × 32 x 64     | 112 x 112 x 32 | 112 x 112 x 64   |
  | Conv dw / s2  | 3 × 3 × 64 dw       | 112 x 112 x 64 | 56 x 56 x 64     |
  | Conv / s1     | 1 × 1 × 64 × 128    | 56 x 56 x 64   | 56 x 56 x 128    |
  | Conv dw / s1  | 3 × 3 × 128 dw      | 56 x 56 x 128  | 56 x 56 x 128    |
  | Conv / s1     | 1 × 1 × 128 × 128   | 56 x 56 x 128  | 56 x 56 x 128    |
  | Conv dw / s2  | 3 × 3 × 128 dw      | 56 x 56 x 128  | 28 x 28 x 128    |
  | Conv / s1     | 1 × 1 × 128 × 256   | 28 x 28 x 128  | 28 x 28 x 256    |
  | Conv dw / s1  | 3 × 3 × 256 dw      | 28 x 28 x 256  | 28 x 28 x 256    |
  | Conv / s1     | 1 × 1 × 256 × 256   | 28 x 28 x 256  | 28 x 28 x 256    |
  | Conv dw / s2  | 3 × 3 × 256 dw      | 28 x 28 x 256  | 14 x 14 x 256    |
  | Conv / s1     | 1 × 1 × 256 × 512   | 14 x 14 x 256  | 14 x 14 x 512    |
  | Conv dw / s1  | 3 × 3 × 512 dw      | 14 x 14 x 512  | 14 x 14 x 512    |
  | Conv / s1     | 1 × 1 × 512 × 512   | 14 x 14 x 512  | 14 x 14 x 512    |
  | Conv dw / s1  | 3 × 3 × 512 dw      | 14 x 14 x 512  | 14 x 14 x 512    |
  | Conv / s1     | 1 × 1 × 512 × 512   | 14 x 14 x 512  | 14 x 14 x 512    |
  | Conv dw / s1  | 3 × 3 × 512 dw      | 14 x 14 x 512  | 14 x 14 x 512    |
  | Conv / s1     | 1 × 1 × 512 × 512   | 14 x 14 x 512  | 14 x 14 x 512    |
  | Conv dw / s1  | 3 × 3 × 512 dw      | 14 x 14 x 512  | 14 x 14 x 512    |
  | Conv / s1     | 1 × 1 × 512 × 512   | 14 x 14 x 512  | 14 x 14 x 512    |
  | Conv dw / s1  | 3 × 3 × 512 dw      | 14 x 14 x 512  | 14 x 14 x 512    |
  | Conv / s1     | 1 × 1 × 512 × 512   | 14 x 14 x 512  | 14 x 14 x 512    |
  | Conv dw / s2  | 3 × 3 × 512 dw      | 14 x 14 x 512  | 7 x 7 x 512      |
  | Conv / s1     | 1 × 1 × 512 × 1024  | 7 x 7 x 512    | 7 x 7 x 1024     |
  | Conv dw / s1  | 3 × 3 × 1024 dw     | 7 x 7 x 1024   | 7 x 7 x 1024     |
  | Conv / s1     | 1 × 1 × 1024 × 1024 | 7 x 7 x 1024   | 7 x 7 x 1024     |
  | AvgPool / s1  | Pool 7 × 7          | 7 x 7 x 1024   | 1 x 1 x 1024     |
  | FC / s1       | 1024 x 1000         | 1 x 1 x 1024   | 1 x 1 x 1000     |
  | Softmax / s1  | Classifier          | 1 x 1 x 1000   | 1 x 1 x 1000     |

  """

  def __init__(self, resolution=224, num_classes=1000, alpha=1):

    super(MobileNetV1, self).__init__()
    assert(resolution % 32 == 0)
    self.first_in_channel = _make_divisible(32 * alpha, 8)
    self.last_out_channel = _make_divisible(1024 * alpha, 8)
    self.net = nn.Sequential(
        Conv2d(3, self.first_in_channel, kernel_size=3, stride=2, padding=1),
        DepthSepConv(32, 64, ksize=3, stride=1, padding=1, alpha=alpha),
        DepthSepConv(64, 128, ksize=3, stride=2, padding=1, alpha=alpha),
        DepthSepConv(128, 128, ksize=3, stride=1, padding=1, alpha=alpha),
        DepthSepConv(128, 256, ksize=3, stride=2, padding=1, alpha=alpha),
        DepthSepConv(256, 256, ksize=3, stride=1, padding=1, alpha=alpha),
        DepthSepConv(256, 512, ksize=3, stride=2, padding=1, alpha=alpha),
        DepthSepConv(512, 512, ksize=3, stride=1, padding=1, alpha=alpha),
        DepthSepConv(512, 512, ksize=3, stride=1, padding=1, alpha=alpha),
        DepthSepConv(512, 512, ksize=3, stride=1, padding=1, alpha=alpha),
        DepthSepConv(512, 512, ksize=3, stride=1, padding=1, alpha=alpha),
        DepthSepConv(512, 512, ksize=3, stride=1, padding=1, alpha=alpha),
        DepthSepConv(512, 1024, ksize=3, stride=2, padding=1, alpha=alpha),
        DepthSepConv(1024, 1024, ksize=3, stride=1, padding=1, alpha=alpha),
        AvgPool2d(kernel_size=resolution // 32, stride=1))

    self.fc = Linear(self.last_out_channel, num_classes)
    self.out = Softmax2d()

  def forward(self, x):
    x = self.net(x)
    x = x.view(-1, self.last_out_channel)
    x = self.fc(x)
    return log_softmax(x, dim=1)


class InvertedResblock(nn.Module):
  """
  docstring for InvertedResblock

  """

  def __init__(self,
               in_channels,
               out_channels,
               ksize=3,
               stride=1,
               padding=1,
               expansion_rate=6,
               multiplier=1):
    super(InvertedResblock, self).__init__()
    self.in_channels = _make_divisible(in_channels * multiplier, 8)
    self.out_channels = _make_divisible(out_channels * multiplier, 8)
    expansion_channel = self.in_channels * expansion_rate
    self.stride = stride
    self.pw_conv1 = Conv2d(in_channels=self.in_channels,
                           out_channels=expansion_channel,
                           kernel_size=1,
                           padding=padding,
                           stride=1)

    self.bn1 = BatchNorm2d(expansion_channel)

    self.dw_conv = Conv2d(in_channels=expansion_channel,
                          out_channels=expansion_channel,
                          kernel_size=3,
                          groups=expansion_channel,
                          padding=padding,
                          stride=stride)

    self.bn2 = BatchNorm2d(expansion_channel)

    self.pw_conv2 = Conv2d(in_channels=expansion_channel,
                           out_channels=self.out_channels,
                           kernel_size=1,
                           padding=padding,
                           stride=1)
    self.bn3 = BatchNorm2d(out_channels)

  def forward(self, x):
    identity = x
    x = self.pw_conv1(x)
    x = self.bn1(x)
    x = relu6(x)
    x = self.dw_conv(x)
    x = self.bn2(x)
    x = relu6(x)
    x = self.pw_conv2(x)
    x = self.bn3(x)
    if self.stride == 1 and self.in_channels == self.out_channels:
      x += identity
    return x


class MobileNetV2(nn.Module):
  """
  [summary]

  MobileNetV2 Body Architecture
  |    Input      |  Operator  |   t   |   c    |   n  |   s   |
  |:-------------:|:----------:|:-----:|:------:|:----:|:-----:|
  |224 x 224 × 3  | conv2d 3x3 |   -   | 32     | 1    | 2     |
  |112 x 112 × 32 | bottleneck |   1   | 16     | 1    | 1     |
  |112 x 112 × 16 | bottleneck |   6   | 24     | 2    | 2     |
  |56 x 56 × 24   | bottleneck |   6   | 32     | 3    | 2     |
  |28 x 28 × 32   | bottleneck |   6   | 64     | 4    | 2     |
  |14 x 14 × 64   | bottleneck |   6   | 96     | 3    | 1     |
  |14 x 14 × 96   | bottleneck |   6   | 160    | 3    | 2     |
  |7 x 7 × 160    | bottleneck |   6   | 320    | 1    | 1     |
  |7 x 7 × 320    | conv2d pw  |   -   | 1280   | 1    | 1     |
  |7 x 7 × 1280   | avgpool    |   -   | -      | 1    | -     |
  |1 x 1 × 1280   | conv2d pw  |   -   | k      | -    | -     |

  """

  def __init__(self, resolution=224, num_classes=1000, multiplier=1):
    super(nn.Module, self).__init__()

    # build first layer
    first_in_channel = _make_divisible(32 * multiplier, 8)
    last_in_channel = _make_divisible(320 * multiplier, 8)
    last_out_channel = _make_divisible(1280 * multiplier, 8)
    last_ksize = resolution // 32
    self.net = nn.Sequential(
        # 224x224x3
        Conv2d(3, first_in_channel, kernel_size=3, stride=2, padding=1),
        # 112 x 112 x 32
        BatchNorm2d(first_in_channel),
        ReLU6(inplace=True),
        InvertedResblock(32, 16, multiplier=multiplier, expansion_rate=1),
        # 112 x 112 x 16
        InvertedResblock(16, 24, multiplier=multiplier, stride=2),
        # 56 x 56 x 24
        InvertedResblock(24, 24, multiplier=multiplier),
        InvertedResblock(24, 32, multiplier=multiplier, stride=2),
        # 28 x 28 x 32
        InvertedResblock(32, 32, multiplier=multiplier),
        InvertedResblock(32, 32, multiplier=multiplier),
        InvertedResblock(32, 64, multiplier=multiplier, stride=2),
        # 14 x 14 x 64
        InvertedResblock(64, 64, multiplier=multiplier),
        InvertedResblock(64, 64, multiplier=multiplier),
        InvertedResblock(64, 64, multiplier=multiplier),
        InvertedResblock(64, 96, multiplier=multiplier),
        # 14 x 14 x 96
        InvertedResblock(96, 96, multiplier=multiplier),
        InvertedResblock(96, 96, multiplier=multiplier),
        InvertedResblock(96, 160, multiplier=multiplier, stride=2),
        # 7 x 7 x 160
        InvertedResblock(160, 160, multiplier=multiplier),
        InvertedResblock(160, 160, multiplier=multiplier),
        InvertedResblock(160, 320, multiplier=multiplier),
        # 7 x 7 x 320
        Conv2d(last_in_channel, last_out_channel, kernel_size=1),
        # 7 x 7 x 1280

        # Global Depthwise Convolution
        # Conv2d(in_channels=last_out_channel,
        #        out_channels=last_out_channel,
        #        kernel_size=last_ksize,
        #        groups=last_out_channel),

        # Global Average Pooling
        AvgPool2d(kernel_size=last_ksize),
        # 1 x 1 x 1280
        Conv2d(last_out_channel, num_classes, kernel_size=1),
        # 1 x 1 x num_classes
        Softmax2d()
    )

  def forward(self, x):
    return self.net(x)
