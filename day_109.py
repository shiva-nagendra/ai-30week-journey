#week 18 day 2
#CNN architecture skeleton

import torch
import torch.nn as nn

class CNNmodel(nn.Module):
    def __init__(self):
        super().__init__()

        #first convolution block
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3
        )

        self.pool = nn.MaxPool2d(2)

        #second convolution block
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3
        )

        

     
