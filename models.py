import torch
import torch.nn as nn
from config import *
from residualBlock import *


"""
    Several units stacked together with the aim of convinience
    - Convolution
    - Instance normalization
    - Non-Linear function
    
    
    We use instance normalization in case of
    developing image2image network
"""
class ConvUnit(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize, stride, pad):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride = stride, padding = pad, bias=False)
        self.norm = nn.InstanceNorm2d(outChannels)
        self.relu = nn.LeakyReLU(0.02)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x



"""
    Encodes picture from domain 'X' to internal feature tensor
"""
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.features = nn.Sequential()
        self.features.add_module('conv1', ConvUnit(3, 64, 7, 2, 3))
        self.features.add_module('conv2', ConvUnit(64, 128, 3, 2, 1))
        self.features.add_module('conv3', ConvUnit(128, 256, 3, 1, 1))


    def forward(self, x):
        x = self.features(x)
        return x



"""
    We assume that inChannels can be diveded by 4
    
    (straightforward barebone implementation)
"""
class SimpleResBlock(nn.Module):
    def __init__(self, inChannels):
        super(SimpleResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(inChannels, inChannels // 4, kernel_size = 1, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(inChannels // 4)
        self.conv2 = nn.Conv2d(inChannels // 4, inChannels // 4, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(inChannels // 4)
        self.conv3 = nn.Conv2d(inChannels // 4, inChannels, kernel_size = 1, stride = 1, padding = 0)
        self.bn3 = nn.BatchNorm2d(inChannels)
        self.relu = nn.LeakyReLU(0.02, inplace=True)
    
    
    def forward(self, x):
        residual = x.to(device = device)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out + residual)
        return out



"""
    Involve several residual blocks
    transforms internal feature tensor
"""
class Converter(nn.Module):
    def __init__(self):
        super(Converter, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module('res1', SimpleResBlock(256))
        self.features.add_module('res2', SimpleResBlock(256))
        self.features.add_module('res3', SimpleResBlock(256))
        self.features.add_module('res4', SimpleResBlock(256))
        self.features.add_module('res5', SimpleResBlock(256))
        self.features.add_module('res6', SimpleResBlock(256))

    def forward(self, x):
        x = self.features(x)
        return x


"""
    Several units stacked together with the aim of convinience
    - Deconvolution
    - Instance normalization
    - Non-Linear function
"""
class DeConvUnit(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize, stride, pad):
        super(DeConvUnit, self).__init__()
        self.convTrans = nn.ConvTranspose2d(inChannels, outChannels, kernelSize,
                                            stride = stride, padding = pad, output_padding = 1, bias=False)
        self.norm = nn.InstanceNorm2d(outChannels)
        self.relu = nn.LeakyReLU(0.02)
        #self.drop = nn.Dropout2d(p=0.4)

    def forward(self, x):
        x = self.convTrans(x)
        x = self.norm(x)
        #x = self.drop(x)
        x = self.relu(x)
        return x


"""
    Decodes internal feature tensor to picture of domain 'Y'
"""
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.features = nn.Sequential()
        """
            input - 64x64 and 256 channels
        """
        self.features.add_module('deconv1', DeConvUnit(256, 128, 3, 2, 1)) # 64x64 -> 128x128
        self.features.add_module('deconv2', DeConvUnit(128, 64, 3, 2, 1)) # 128x128 -> 256x256
        self.features.add_module('conv3', ConvUnit(64, 3, 7, 1, 3))       # 256x256 -> 256x256

            
    def forward(self, x):
        x = self.features(x)
        return x



"""
    Full Transforming Model
    - Encoder
    - Converter
    - Decoder
"""
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.features = nn.Sequential()
        
        self.features.add_module('encoder', Encoder())
        self.features.add_module('converter', Converter())
        self.features.add_module('decoder', Decoder())
    
    
    def forward(self, x):
        x = self.features(x)
        return x



"""
    Distinguishes generated images from real
    'PatchGAN' imlementation
    the most efficient patch size is considerd as 70x70
"""
class PatchGAN(nn.Module)
    def __init__(self):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module('conv1', ConvUnit(3, 64, 4, 2, 0))         # 70x70 -> 34x34
        self.features.add_module('conv2', ConvUnit(64, 128, 4, 2, 0))       # 34x34 -> 16x16
        self.features.add_module('conv3', ConvUnit(128, 256, 4, 2, 1))      # 16x16 -> 8x8
        self.features.add_module('conv4', ConvUnit(256, 512, 4, 2, 3))      # 8x8 -> 4x4
        self.features.add_module('conv5', ConvUnit(512, 1, 4, 1, 0))        # 4x4 -> 1x1
        self.features.add_module('activation', nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        return x
