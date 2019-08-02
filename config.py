import os as os
from os import listdir
from os.path import isfile, join
import json
import torch


device = "cpu"
dtype = torch.float32
imageShape = (256,256,3)
replayMemorySize = 50


epochs = 30
iterations = 1000
learningRate = 0.0002
