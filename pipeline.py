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

