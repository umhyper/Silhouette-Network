import torch
import torch.nn as nn

class SlimBatchNorm2d(nn.Sequential):
    def __init__(self, in_ch, bn_epsilon=0.001):
        super(SlimBatchNorm2d, self).__init__(
                nn.BatchNorm2d(in_ch, eps=bn_epsilon)
        )

        
# "Padding : valid = witout padding"
# "Padding: same = with zero padding"

class SlimConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(SlimConv2d, self).__init__()
        self.slim_conv = nn.Sequential(
                #nn.ReflectionPad2d(padding),
                nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
                SlimBatchNorm2d(out_ch),
                nn.ReLU()
        )

    def forward(self, x):
        return self.slim_conv(x)


class SlimConv2dTranspose(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1):
        super(SlimConv2dTranspose, self).__init__()
        self.slim_conv2d_transpose = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride,padding=padding),
                SlimBatchNorm2d(out_ch),
                nn.ReLU()
        )

    def forward(self, x):
        return self.slim_conv2d_transpose(x)


class SlimFullyConnected(nn.Module):

    def __init__(self, in_feature, out_feature):
        super(SlimFullyConnected, self).__init__()
        self.slim_fully_connected = nn.Sequential(
                nn.Linear(in_feature, out_feature),
                nn.BatchNorm1d(out_feature),
                nn.ReLU()
        )

    def forward(self, x):
        return self.slim_fully_connected(x)


class ConvMaxpool(nn.Module):
    """ simple conv + max_pool """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(ConvMaxpool, self).__init__()
        self.conv = SlimConv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.max_pool2d(self.conv(x)) 