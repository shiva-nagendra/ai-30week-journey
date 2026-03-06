#week 18 day 2
#CNN architecture skeleton

import torch
import torch.nn as nn

class CNNmodel(nn.Module):
    def __init__(self):
        super().__init__()

        #first convolution block
