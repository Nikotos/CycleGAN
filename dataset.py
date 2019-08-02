import torch
import random
from random import shuffle
import config
import pickle
import numpy as np

def saveToFile(object, filename):
    with open(filename, "wb") as file:
        pickle.dump(object, file)

def loadFromFile(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


class DataLoader:
    def __init__(self):
        self.storage = []
    
    def __len__(self):
        return len(self.storage)

    """
        It is compulsory to add torch tensors here
    """
    def add(self, element):
        self.storage.append(element)
    
    def getIndexed(self, index):
        return self.storag[index]

    def get(self):
        index = np.random.randint(0, len(self.storage))
        return self.storage[index]



"""
    Some reinforcement learning hacks
"""

class ReplayMemory:
    def __init__(self, size):
        self.data = []
        self.size = size
        self.currentPosition = 0
    
    def __len__(self):
        return len(self.dataset)
    
    def add(self, element):
        if (len(self.data) < self.size):
            self.data.append(element)
        else:
            self.data[self.currentPosition] = element
            self.currentPosition = (self.currentPosition + 1) % self.size

    def get(self):
        index = np.random.randint(0, self.size)
        return self.data[i]


