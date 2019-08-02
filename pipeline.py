from config import *
from dataset import *
from model import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2


"""
    initializing:
    - 2 generators and
    - 2 correspondinf discriminators
    - loading data loaders
    - setting up optimizers
"""
transformerForward = Transformer()
transformerBackward = Transformer()

discriminatorA = PatchGAN()
discriminatorB = PatchGAN()

dataLoaderA = loadFromFile("dataLoaderA.pkl")
dataLoaderB = loadFromFile("dataLoaderB.pkl")

optimizerForward = optim.Adam(transformerForward.parameters(), lr = learningRate, betas = (beta1, 0.999))
optimizerBackward = optim.Adam(transformerBackward.parameters(), lr = learningRate, betas = (beta1, 0.999))

lastFewGenSamples = ReplayMemory(50)



"""
    Main Pipeline
    We need to bulid general logic of training network
    with both adversial and cycle consistency loss
    
    1) perform the real picture pass through discriminators
    2)
"""

for e in range(config.epochs):
    """
        Epoch setup, such as lr decay
    """
    for i in range(config.iterations):
        





