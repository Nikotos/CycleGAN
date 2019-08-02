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
    
    1) perform the real picture pass through discriminators (only one part of loss)
    
    2) perform fake image pass through discriminators
       generate one sample (A->B and B->A) (detach it) and add it to memory and then
       take picture from replay memory (to decrease model oscillations)
       
    3) update both discriminator weights
    
    4) Genetare both images, A->B and B->A, calculate both adversarial loss
       using both discriminators, store it
       
    5) Complete the cycle, generating A_cycle from (A->B) and B_cycle from (B->A)
       calsulate cycle consistency loss, multiply by hyperparameter
       (mean abs loss)
       
    6) stack two losses together and update weights of both transformers
"""

for e in range(config.epochs):
    """
        Epoch setup, such as lr decay
    """
    for i in range(config.iterations):
        





