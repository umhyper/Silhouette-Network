  class ResnetkBlock(nn.module):
        """
        Input    shape: (1, 32, 32, 32)
        Output shape: (1, 32, 32, 32)
        
        the shape remains the same --> simplify
        in_c, out_c --> channel  (2-->1)
        
        """
        def __init__(self, ch, kernel_size=3, scale=1.0, activation_fn=tf.nn.relu):
            super(ResnetkBlock, self).__init__()
    
            self.conv1 = slimconv2d(ch, ch/2, kernel_size=1, stride=1)     # tower0  (1, 32, 32, 16)
            self.conv2 = slim.conv2d(ch, ch/2, kernel_size=1, stride=1)   # tower1 (1, 32, 32, 16)
            self.conv3 = slim.conv2d(ch/2, ch/2, kernel_size= [1, kernel_size], stride=1) # tower1 (1, 32, 32, 16)
            self.conv4 = slim.conv2d(ch/2, ch/2, kernel_size= [kernel_size, 1], stride=1) # tower1 (1, 32, 32, 16)
            self.conv5 = conv2d(ch/2, ch, kernel_size=1, stride=1)  # mixup (1, 32, 32, 32)
            self.scale = scale
            self.relu = nn.ReLU(inplace=True)
   
        def forward(self, x):
    
            tower0 = self.conv1(x)
            tower1 = self.conv2(x)
            tower1 = self.conv3(tower1)
            tower1 = self.conv4(tower1)
            
            mixed = torch.cat((tower0, tower1), 1)  # mixed = self.concat(axis=-1, values=[tower0, tower1]) (1, 32, 32, 32)
            mixup = self.conv2d(mixed)

            out += mixup * self.scale
            out = self.relu(out)
            
        return out


