import torch.nn as nn
import torch
from model.utils import SlimConv2d, SlimConv2dTranspose

class DPN(nn.Module):

    """
    Generate guidance map

    """
    def __init__(self, in_ch, kernel_size=3):
        super(DPN, self).__init__()

        self.down1 = SlimConv2d(in_ch, 32, kernel_size, stride=1)
        self.down2 = SlimConv2d(32, 32, kernel_size, stride=2)
        self.down3 = SlimConv2d(32, 64, kernel_size, stride=2)
        self.down4 = SlimConv2d(64, 128, kernel_size, stride=2)

        # Transpose  kernel size check, change from 2-->4
        self.up1 = SlimConv2dTranspose(128, 128, 4, stride=2)
        self._p3 = SlimConv2d(64, 128, kernel_size, stride=1)

        self.up2 = SlimConv2dTranspose(128, 64, 4, stride=2)
        self._p2 = SlimConv2d(32, 64, kernel_size, stride=1)

        self.up3 = SlimConv2dTranspose(64, 32, 4, stride=2)
        self._p1 = SlimConv2d(32, 32, kernel_size, stride=1)

        self.pad = nn.ReflectionPad2d(3)
        self.conv = nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # (p1, p2, p3, p4) = (128, 64, 32, 16) --> Resolution
        # (p1, p2, p3, p4) = (32, 32, 64, 128) --> Channel

        p1 = self.down1(x)
        p2 = self.down2(p1)
        p3 = self.down3(p2)
        p4 = self.down4(p3)    # (?, 16, 16, 128)
       
        d1 = self.up1(p4)
        p3 = self._p3(p3)
#         print(d1.shape)
#         print(p3.shape)
        d1 = p3 + d1

        d2 = self.up2(d1)
        p2 = self._p2(p2)
        d2 += p2

        d3 = self.up3(d2)
        p1 = self._p1(p1)
        d3 += p1
      
        d4 = self.conv(self.pad(d3))
        # print(d4.shape)  # torch.Size([4, 3, 320, 320])
        fake_depth = self.sigmoid(d4)

	    # Not sure if the return is correct
        return d1, d2, d3, fake_depth



