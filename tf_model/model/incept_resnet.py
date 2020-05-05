import torch
import torch.nn as nn

from model.utils import SlimConv2d, ConvMaxpool, SlimConv2dTranspose, SlimFullyConnected
import math

class PullOut8(nn.Module):
    
    """
        Input shape x: (batch_size, 8, 8, 128)
        Out dim = Bx(Jx3) 
    """
    def __init__(self, in_ch, out_ch):
        super(PullOut8, self).__init__()
        self.conv_maxpool1 = ConvMaxpool(in_ch, 256)
        self.conv_maxpool2 = ConvMaxpool(256, 512)
        self.conv1 = SlimConv2d(512, 1024)  # Note that : valid padding (original)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc =  SlimFullyConnected(36864, 63)  # fc_num = 512*2 = 1024

    def forward(self, x): 
        """ supposed to work best with 8x8 input """
        x = self.conv_maxpool1(x) # (? ,4, 4, 256)
        x = self.conv_maxpool2(x) # (?, 2, 2, 512)
        x = self.conv1(x)
        
        x = self.flatten(x)
        x = self.dropout(x)  # flatten + dropout
        x = self.fc(x) 
  
        return x

    
class ResnetkBlock(nn.Module):
        """
        Input    shape: (1, 32, 32, 32)
        Output shape: (1, 32, 32, 32)
        
        the shape remains the same --> simplify
        in_c, out_c --> channel  (2-->1)
        
        """
        def __init__(self, ch, kernel_size=3, scale=1.0):
            super(ResnetkBlock, self).__init__()
           
            in_c = ch
    
            self.conv0_1 = SlimConv2d(in_c, ch//2, kernel_size=1, stride=1, padding=0)   # tower0  (1, 32, 32, 16)
            self.conv1_1 = SlimConv2d(in_c, ch//2, kernel_size=1, stride=1, padding=0)   # tower1 (1, 32, 32, 16)
            
            # Make some changes: from [3, 1] and [1, 3]  --> kernel_size = 3
            self.conv1_2 = SlimConv2d(ch//2, ch//2, kernel_size= (1, 3), stride=1, padding=0) # tower1 (1, 32, 32, 16)
            self.conv1_3 = SlimConv2d(ch//2, ch//2, kernel_size= (3, 1), stride=1, padding=1) # tower1 (1, 32, 32, 16)
            
            self.conv_mix = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)  # mixup (1, 32, 32, 32)
            self.scale = scale
            self.relu = nn.ReLU(inplace=True)
            
        def forward(self, x):

            tower0 = self.conv0_1(x)
            tower1 = self.conv1_1(x)
            tower1 = self.conv1_2(tower1)
            tower1 = self.conv1_3(tower1)
    
            mixed = torch.cat((tower0, tower1), 1)  # mixed = self.concat(axis=-1, values=[tower0, tower1]) (1, 32, 32, 32)
            mixup = self.conv_mix(mixed)
    
            out = self.relu(mixup * self.scale)

            return out     
        
        