import torch
from torch import nn
from torch.nn import functional as F

from models.dilated_resnet import dilated_resnet18
#from dilated_resnet import dilated_resnet18

"""
BackBone: DilatedResNet18
"""

__all__ = ["DeepLabV3Plus"]


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ASPPConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, dilation):
        modules = [
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch, atrous_rates):
        super(ASPP, self).__init__()
        out_ch = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_ch, out_ch, rate1))
        modules.append(ASPPConv(in_ch, out_ch, rate2))
        modules.append(ASPPConv(in_ch, out_ch, rate3))
        modules.append(ASPPPooling(in_ch, out_ch))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabHead(nn.Module):
    def __init__(self, nclass, low_ch, **kwargs):
        super(DeepLabHead, self).__init__()
        self.low_block = ConvBNReLU(low_ch, 48, 3, padding=1)
        self.block = nn.Sequential(
            ConvBNReLU(304, 256, 3, padding=1),
            nn.Dropout(0.5),
            ConvBNReLU(256, 256, 3, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(256, nclass, 1))

    def forward(self, x, low):
        size = low.size()[2:]
        low = self.low_block(low)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return self.block(torch.cat([x, low], dim=1))


class DepthHead(nn.Module):
    def __init__(self, low_ch, **kwargs):
        super(DepthHead, self).__init__()
        self.low_block = ConvBNReLU(low_ch, 48, 3, padding=1)
        self.block = nn.Sequential(
            ConvBNReLU(304, 256, 3, padding=1),
            ConvBNReLU(256, 256, 3, padding=1))

    def forward(self, x, low):
        size = low.size()[2:]
        low = self.low_block(low)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return self.block(torch.cat([x, low], dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, nclass, pretrained=False):
        super(DeepLabV3Plus, self).__init__()
        self.nclass = nclass

        resnet = list(dilated_resnet18(pretrained).children())
        self.conv1 = nn.Sequential(*resnet[:4])
        self.down1 = nn.Sequential(*resnet[4])
        self.down2 = nn.Sequential(*resnet[5])
        self.down3 = nn.Sequential(*resnet[6])
        self.down4 = nn.Sequential(*resnet[7])

        self.aspp = ASPP(512, [6, 12, 18])

        self.seg_head = DeepLabHead(self.nclass, low_ch=128)

        self.depth_head = DepthHead(low_ch=128)
        self.depth_block = nn.Sequential(
            ConvBNReLU(256, 256, 3, padding=1),
            nn.Dropout(0.5),
            ConvBNReLU(256, 256, 3, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(256, 1, 1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.size()[2:]
        x = self.conv1(x)
        x = self.down1(x)  # 1/4 input_size
        low = self.down2(x)  # 1/8 input_size
        x = self.down3(low)  # 1/8 input_size
        x = self.down4(x)  # 1/8 input_size
        x = self.aspp(x)

        x_seg = self.seg_head(x, low)
        x_seg = F.interpolate(x_seg, size, mode='bilinear', align_corners=False)

        x_depth = self.depth_head(x, low)
        x_depth = F.interpolate(x_depth, size, mode='bilinear', align_corners=False)
        x_depth = self.depth_block(x_depth)

        return { 'mask'   : x_seg, 
                 'depth' : x_depth }


if __name__ == '__main__':
    import time
    nclass = 3
    a = torch.ones(2, 3, 320, 320)
    model = DeepLabV3Plus(nclass)
    st = time.time()
    out_dic = model(a)
    print(time.time() - st)
    print(out_dic['seg'].shape)
    print(out_dic['depth'].shape)
