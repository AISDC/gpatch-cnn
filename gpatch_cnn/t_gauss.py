import torch
import random
import numpy as np
import torchvision

def singleGauss(size,amp,sigma_x,sigma_y,x0,y0):
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

def multiGauss(n,size):
    size = 11
    z = torch.zeros(size=(size,size))
    for i in range(n):
        z_i   = singleGauss(size,AmpRand(),SigmaRand(),SigmaRand(),initRand(),initRand())
        z    += z_i
    return z

def write_images(z,img_name):
    torchvision.utils.save_image(z,img_name)

def genData(patchSize,trainSize,n_peaks): 
    for i in range(trainSize):
        mult=multiGauss(n_peaks)
    for i in range(trainSize):
        single=singleGauss(SigmaRand(),SigmaRand(),initRand(),initRand(),initRand())
    return(single,mult)
