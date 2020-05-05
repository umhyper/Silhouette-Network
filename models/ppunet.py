import torch.nn as nn
import torch.nn.functional as F
import torch
from models.dilated_resnet import dilated_resnet18
#from dilated_resnet import dilated_resnet18

class BaseConv(nn.Module):
    def __init__(self,in_plane,out_plane):
        super(BaseConv,self).__init__()
        self.baseConv = nn.Sequential(
            nn.Conv2d(in_plane, out_plane, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_plane),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_plane, out_plane, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_plane),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.baseConv(x)
        return x


class PCBR(nn.Module):
    """
    AveragePooling => Conv => ReLU 
    """
    def __init__(self, out_size, in_plane, out_plane):
        super(PCBR, self).__init__()
        self.pcbr = nn.Sequential(
                nn.AdaptiveAvgPool2d(out_size),
                nn.Conv2d(in_plane, out_plane, kernel_size=1), 
                nn.BatchNorm2d(out_plane),
                nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.pcbr(x)


class PyramidPooling(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(PyramidPooling, self).__init__()
        self.pool1 = PCBR(out_size=1, in_plane=in_plane, out_plane=out_plane)
        self.pool2 = PCBR(out_size=2, in_plane=in_plane, out_plane=out_plane)
        self.pool3 = PCBR(out_size=4, in_plane=in_plane, out_plane=out_plane)
        self.pool4 = PCBR(out_size=6, in_plane=in_plane, out_plane=out_plane)

    def forward(self, x):
        size = x.shape[2:]
        x1 = F.interpolate(self.pool1(x), size=size, mode='bilinear')
        x2 = F.interpolate(self.pool2(x), size=size, mode='bilinear')
        x3 = F.interpolate(self.pool3(x), size=size, mode='bilinear')
        x4 = F.interpolate(self.pool4(x), size=size, mode='bilinear')
        out = torch.cat([x1,x2,x3,x4,x], dim=1)
        return out


class UpConv(nn.Module):
    def __init__(self, in_plane, add_plane, out_plane):
        # add_plane = plane(channel) of features from intermediate layer in Encoder (=channel of x2 in this forward function)
        super(UpConv,self).__init__()
        self.up = nn.Sequential(
                nn.ConvTranspose2d(in_plane, out_plane, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
        )
        self.conv = BaseConv(out_plane+add_plane, out_plane)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        size = x1.shape[-2:]
        x2 = F.interpolate(x2, size=size, mode='bilinear')
        x = torch.cat([x1,x2], dim=1)
        return self.conv(x)



class PPUNet(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(PPUNet, self).__init__()

        resnet = list(dilated_resnet18(pretrained).children())
        self.conv1 = nn.Sequential(*resnet[:4])
        self.down1 = nn.Sequential(*resnet[4]) 
        self.down2 = nn.Sequential(*resnet[5]) 
        self.down3 = nn.Sequential(*resnet[6]) 
        self.down4 = nn.Sequential(*resnet[7])

        self.pyramidpool = PyramidPooling(in_plane=512, out_plane=128)
        self.up1 = UpConv(1024, 256, 512)
        self.up2 = UpConv(512, 128, 256)
        self.up3 = UpConv(256, 64, 128)
        self.score = BaseConv(128, num_classes)


    def forward(self,x):
        x = self.conv1(x)
        x1 = self.down1(x) #1/4 input_size
        x2 = self.down2(x1) #1/8 input_size
        x3 = self.down3(x2) #1/8 input_size
        x = self.down4(x3) #1/8 input_size
        x = self.pyramidpool(x)
        x = self.up1(x,x3) #1/4 input_size
        x = self.up2(x,x2) #1/2 input_size
        x = self.up3(x,x1) #input_size
        x = self.score(x)

        return x

if __name__ == '__main__':
    import torch
    a = torch.ones(2,3,320,320)
    model = PPUNet(1)
    print(model(a).shape)
