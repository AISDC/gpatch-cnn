import os
import time
import copy
import torch
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import torchvision
import torch.nn as nn
import torch.optim as optim
from skimage import transform
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from PIL import Image as im

def singleGauss(amp,sigma_x,sigma_y,x0,y0):
    size = 2048
    x    = torch.linspace(-1, 1, size)
    y    = torch.linspace(-1, 1, size)
    x, y = torch.meshgrid(x, y)
    z    = (1/(2*np.pi*sigma_x*sigma_y) * amp*np.exp(-((x-x0)**2/(2*sigma_x**2)
       		+ (y-y0)**2/(2*sigma_y**2))))
    return z

def AmpRand():
    return random.random()

def SigmaRand():
    return random.random()*0.5
    
def initRand():
    return random.random()*0.5

def multiGauss(n):
    size = 2048
    z = torch.zeros(size=(size,size))
    for i in range(n):
        z_i   = singleGauss(AmpRand(),SigmaRand(),SigmaRand(),initRand(),initRand())
        z    += z_i
    return z

def torchMax(z):
    return torch.max(z)

def write_images(z,img_name):
    z = transforms.Grayscale() 
    torchvision.utils.save_image(z,img_name)

def genTrainData(trainSize): 
    if os.path.exists('./data/train/two') is False:
        os.mkdir('two')
    for i in range(trainSize):
        test_mult=multiGauss(2)
        write_images(test_mult,'./two/img%i.png' %i)
    
    if os.path.exists('./data/train/one') is False:
        os.mkdir('one')
    for i in range(trainSize):
        test=singleGauss(SigmaRand(),SigmaRand(),initRand(),initRand())
        write_images(test,'./one/img%i.png' %i)

def genValData(valSize): 
    if os.path.exists('./data/val/two') is False:
        os.mkdir('./data/val/two')
    for i in range(valSize):
        test_mult=multiGauss(2)
        write_images(test_mult,'./data/val/two/img%i.png' %i)
    
    if os.path.exists('./data/val/one') is False:
        os.mkdir('./data/val/one')
    for i in range(valSize):
        test=singleGauss(SigmaRand(),SigmaRand(),initRand(),initRand())
        write_images(test,'./data/val/one/img%i.png' %i)

