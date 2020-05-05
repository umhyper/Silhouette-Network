""" Estimate 3D Hand Pose through binary Mask
Author: Wayne Lee
"""
import torch.nn as nn
import torch.nn.functional as F

from model.incept_resnet import  ResnetkBlock, PullOut8
from model.DPN import DPN
from model.utils import SlimConv2d, ConvMaxpool, SlimConv2dTranspose

class Hourglass(nn.Module):
    """
        Input shape: (1, 32, 32, 64)
        Out: (1,32,32,64)
        2 Times Error Still Exist
    """
    def __init__(self, ch, n):
        super(Hourglass, self).__init__()
        
        self.n = n
        self.hg = self.make_hg(ch, n)  
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
    def make_hg(self, ch, n):
        hg = []
        for i in range(n):
            m = []
            for j in range(3):
                m.append(ResnetkBlock(ch))

            if i==0:
                m.append(ResnetkBlock(ch))
            hg.append(nn.ModuleList(m))    
        
        return nn.ModuleList(hg)
    
#         self.resnet_k1 = ResnetkBlock(ch)
#         self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
#         self.resnet_k2 = ResnetkBlock(ch)
#         self.conv1 = SlimConv2d(ch, ch*2, kernel_size=1, stride=1)
#         self.conv2 = SlimConv2d(ch, ch, kernel_size=1, stride=1)
#         self.resnet_k3 = ResnetkBlock(ch)
#         self.conv_transpose = SlimConv2dTranspose(ch, ch, kernel_size=3, stride=2)
        
    def _hour_glass_forward(self, x, n):
        """ input shape: (1, 32, 32, 64) """

        upper0 = self.hg[n-1][0](x)     

        lower0 = F.max_pool2d(x, 2, stride=2) # 16,16,64         8,8,64
        lower0 = self.hg[n-1][1](lower0)
        
        if n > 1:
            lower1 = self._hour_glass_forward(lower0, n-1)
        else:
            lower1 = self.hg[n-1][2](lower0)

        lower2 = self.hg[n-1][2](lower1)
        upper1 = F.interpolate(lower2, scale_factor=2)

        out = upper0 + upper1
        
        return out
    
#         upper0 = self.resnet_k1(x) # upper0 (1, 32, 32, 64) , 16,16,128
      
#         lower0 = self.max_pool1(x) # 16,16,64            8,8,64
#         lower0 = self.resnet_k2(lower0) #16,16,64     8,8,128
#         lower0 = self.conv1(lower0) #16,16,128        8,8,256

#         if self.ntimes > 1:
#             lower1 = self._hour_glass_forward(lower0, n-1) #16,16,128
#         else:
#             lower1 = lower0

#         lower1 = self.conv2(lower1) # 8,8,128    

#         lower2 = self.resnet_k3(lower1)
#         upper1 = self.conv_transpose(lower2) 
#         out = upper0 + upper1
#         return out
      
    def forward(self, x):
        return self._hour_glass_forward(x, self.n)


class RPN(nn.Module):
    ### Residual Module ###

    def __init__(self, num_feature, is_rgb, hg_repeat=1, num_joints=21):
        super(RPN, self).__init__()
        
        self.hg_repeat = hg_repeat
        
        if is_rgb:
            self.conv1 = SlimConv2d(3, 8, kernel_size=3)
        else:
            self.conv1 = SlimConv2d(1, 8, kernel_size=3)
            
        self.conv_maxpool1 = ConvMaxpool(8, 16)
        self.conv_maxpool2 = ConvMaxpool(16, 32)
        self.resnet_k1 = ResnetkBlock(32)
        self.conv2 = SlimConv2d(32, num_feature, kernel_size=1)

        self.hg_net = Hourglass(num_feature, n=hg_repeat )
        self.resnet_k2 = ResnetkBlock(num_feature)

        self.conv3 = nn.Conv2d(num_feature, num_joints*3, kernel_size=1)
        self.conv4 = SlimConv2d(num_joints*3, num_feature, kernel_size=1, padding=0)
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_maxpool3 = ConvMaxpool(64, 128)
        self.pullout8 = PullOut8(128, 63)

    def forward(self, x):
        # Input: (?, 128, 128, 3)
        
        stg128 = self.conv1(x)  # (?, 128, 128, 8)
        stg64 = self.conv_maxpool1(stg128)  # (? , 64, 64, 16)
        stg32 = self.conv_maxpool2(stg64)  # (?, 32, 32, 32), scope = 'stage64_image'

        #scope = 'stage32_pre'
        stg32 = self.resnet_k1(stg32)
        out = self.conv2(stg32) #(?, 32, 32, 64)

        for hg in range(self.hg_repeat):

            ## TODO: we might replace 'hourglass' to other latest framework
            branch0 = self.hg_net(out)
            branch0 = self.resnet_k2(branch0)

            # Multiply bt 3 here, styled Map becomes 63
            heat_maps = self.conv3(branch0)  # (? 32, 32, 63)
            branch1 = self.conv4(heat_maps)

            out += branch0 + branch1

        ##  TODO: check max pool only

        out = self.max_pool2d(out)  # (?, 16, 16, 64)
        out = self.conv_maxpool3(out) # (?, 8, 8, 128)
        out = self.pullout8(out) # (?, 63)

        return out

class SilhouetteNet(nn.Module):
    """
    End-to-end 3D hand pose estimation from a single binary mask
    This class use clean_depth (128, 128, 1), clean_binary(128, 128, 1)
    Plus Multiview data (128, 128, 3)
    'MV' stands for Multi-View

    ### Size of heatmap ###
    # num_feature = 32
    # mv = multi-view = 3

    """

    def __init__(self, mv=3, num_feature=64, hg_repeat=2, is_rgb=False):
        super(SilhouetteNet, self).__init__()
        
        if  is_rgb:
            self.dpn = DPN(in_ch=3)
        else:
            self.dpn = DPN(in_ch=1)
            
        self.conv1 = SlimConv2d(32, 21*3, stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  ## add max pool padding
        self.conv2 = SlimConv2d(64,  21*3, kernel_size=3, stride=1)
        
        ## TODO: Check slim.max_pool2d
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  ## add max pool padding
        self.conv3 = SlimConv2d(128, 21*3, kernel_size=3, stride=1)
        self.conv4 = SlimConv2d(21*3, 21*3, kernel_size=3, stride=1)

        self.rpn = RPN(num_feature, is_rgb)

    def forward(self, x):

        d1, d2, d3, content_code = self.dpn(x)

        br0 = self.conv1(d3)
        br0 = self.max_pool1(br0)  ## (?, 64, 64, 63)
        br1 = self.conv2(d2)  ## (?, 64, 64, 63), scope = 'hmap64'
        out = br0 + br1
        
        out = self.max_pool2(out)  ##scope = 'hmap32', (32, 32, 63)
        br2 = self.conv3(d1)  ## (?, 32, 32, 63) scope = 'mv_hmap32'
        out = out + br2
        
        guidance = self.conv4(out)  ## (?, 32, 32, 63)
        pose = self.rpn(x)
        
        return guidance, pose

if __name__ == '__main__':
    import torch 
    x = torch.ones(8, 1, 320, 320)
    model = SilhouetteNet()
    print(model(x)[0].shape)   ## 1/4 resolution
    print(model(x)[1].shape)   ## Hand Pose, 63  = 21 * 3

