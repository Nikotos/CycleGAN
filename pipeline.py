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
    - setting up criterions
"""
transformerForward = Transformer()
transformerBackward = Transformer()

discriminatorA = PatchGAN()
discriminatorB = PatchGAN()

dataLoaderA = loadFromFile("dataLoaderA.pkl")
dataLoaderB = loadFromFile("dataLoaderB.pkl")

optimizerTransForward = optim.Adam(transformerForward.parameters(), lr = learningRate, betas = (beta1, 0.999))
optimizerTransBackward = optim.Adam(transformerBackward.parameters(), lr = learningRate, betas = (beta1, 0.999))

optimizerDiscA = optim.Adam(transformerForward.parameters(), lr = learningRate, betas = (beta1, 0.999))
optimizerDiscB = optim.Adam(transformerForward.parameters(), lr = learningRate, betas = (beta1, 0.999))

memoryFakeA = ReplayMemory(replayMemorySize)
memoryFakeB = ReplayMemory(replayMemorySize)

bceLoss = nn.BCELoss()


"""
    in some studies people use perceptual loss with vgg-11 or vgg-19
"""
def cycleConsistencyLoss(x,y):
    return np.mean(np.abc(x - y))


"""
    as it was stated in 'ganhacks' noisy labels encourages model
    to better convergence
    reference [https://github.com/jaingaurav3/GAN-Hacks]
"""

def noisyRealLabel():
    return torch.tensor(np.random.uniform(0.8, 1.2))

def noisyFakeLabel():
    return torch.tensor(np.random.uniform(0.0, 0.3))


"""
    Main Pipeline
    We need to bulid general logic of training network
    with both adversial and cycle consistency loss
    
    1) perform the real picture pass through discriminators (only one part of loss)
    
    2) perform fake image pass through discriminators
       generate one sample (A->B and B->A) (detach it) and add it to memory and then
       take picture from replay memory (to decrease model oscillations)
       
    3) update both discriminator weights
    
    4) Using already generated images, A->B and B->A, calculate both adversarial loss
       using both discriminators, store it
       
    5) Complete the cycle, generating A_cycle from (A->B) and B_cycle from (B->A)
       calsulate cycle consistency loss, multiply by hyperparameter
       (mean abs loss)
       
    6) stack two losses together and update weights of both transformers
"""


"""
    preparing memory of generated images
    to use them to train discriminator
"""
def prepareMemory(transformerNet, memory, dataLoader):
    for i in range(config.replayMemorySize):
        element1 = dataLoader.get()
        elementFake2 = transformerNet(element1)
        memory.add(elementFake2)

"""
    1-st Stage
    performs real picture pass
    returns corresponding loss
"""
def realPicturePass(discriminatorNet, imageOriginal):
    discriminatorNet.zero_grad()
    label = noisyRealLabel()
    prediction = discriminatorNet(imageOriginal).view(-1)
    loss = bceLoss(prediction, label)
    return loss


"""
    2-st Stage
    performs transformed picture pass to train discriminator
    taeks fake image from memory
    returns corresponding loss and generated image
"""
def fakePicturePass(transformerNet, dataLoader, memory, discriminator):
    # firstly create image and then add it to memory
    image = dataLoader.get()
    imageFake = transformerNet(image)
    memory.add(imageFake.detach())

    # then take one image from memory and calculate loss on it
    imageFakeOld = memory.get()
    label = noisyFakeLabel()
    prediction = discriminator(imageFakeOld).view(-1)
    loss = bceLoss(prediction, label)
    return loss, imageFake


"""
    4-th stage
    generate image with transformer and evaluate it with discriminator
    it is important to remind that generator 'thinks' that generated images are real
    (it is part of Nash minimax game)
"""
def generatedImageEvaluation(imageGenerated, discriminator):
    label = noisyRealLabel()
    prediction = discriminator(imageGenerated).view(-1)
    loss = bceLoss(prediction, label)
    return loss


"""
    5-th stage
    competing cycle
    - generating image back
    - calculate cycle loss between original image and generated back
"""
def cycleConsistencyEvaluation(imageOriginal, imageGenerated, transformerBackNet)
    imageGeneratedBack = transformerBackNet(imageGenerated)
    loss = cycleConsistencyLoss(imageOriginal, imageGeneratedBack)
    return loss


prepareMemory(transformerForward, memoryFakeB, dataLoaderA)
prepareMemory(transformerBackward, memoryFakeA, dataLoaderB)

discriminatorA.train()
discriminatorB.train()
transformerForward.train()
transformerBackward.train()

for e in range(config.epochs):
    """
        Epoch setup, such as lr decay
    """
    for i in range(config.iterations):
        # 1-st stage
        imageOriginalA = dataLoaderA.get().to(device = device, dtype = dtype)
        imageOriginalB = dataLoaderB.get().to(device = device, dtype = dtype)
        
        lossRealDiscA = realPicturePass(discriminatorA, imageOriginalA)
        lossRealDiscB = realPicturePass(discriminatorB, imageOriginalB)

        # 2-nd stage
        lossFakeDiscA, imageFakeB = fakePicturePass(transformerForward, dataLoaderA, memoryFakeB, discriminatorB)
        lossFakeDiscB, imageFakeA = fakePicturePass(transformerBackward, dataLoaderB, memoryFakeA, discriminatorA)

        # 3-rd stage
        totalDLossA = lossRealDiscA + lossFakeDiscA
        totalDLossB = lossRealDiscB + lossFakeDiscB

        totalDLossA.backward()
        totalDLossB.backward()

        optimizerDiscA.step()
        optimizerDiscB.step()


        # 4-th stage
        lossGenAdversarialForward = generatedImageEvaluation(imageFakeB, discriminatorB)
        lossGenAdversarialBackward = generatedImageEvaluation(imageFakeA, discriminatorA)


        # 5-th stage
        lossCycleA = cycleConsistencyEvaluation(imageOriginalA, imageFakeB, transformerBackward)
        lossCycleB = cycleConsistencyEvaluation(imageOriginalB, imageFakeA, transformerForward)


        # 6-th stage
        totalGenLossA = lossGenAdversarialForward






