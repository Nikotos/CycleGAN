import torch
import torch.nn as nn
from config import *
from residualBlock im *


"""
    Encodes picture from domain 'X' to internal feature tensor
"""
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module('conv1', HorseUnit(100, d * 4, 4, 1, 0))


    def forward(self, x):
        x = self.features(x)
        return x



"""
    Involve several residual blocks
    transforms internal feature tensor
"""
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module('residual1', ResidualBlock(100, d * 4, 4, 1, 0))

    def forward(self, x):
        x = self.features(x)
        return x


"""
    Decodes internal feature tensor to picture of domain 'Y'
"""
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.features = nn.Sequential()

            
    def forward(self, x):
        x = self.features(x)
        return x




"""
    Distinguishes generated images from real
    'PatchGAN' imlementation
"""
class Discriminator
    def __init__(self):
        super(Decoder, self).__init__()
        


    def forward(self, x):
        x = self.features(x)
        return x
