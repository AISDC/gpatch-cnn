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
    return random.random()*0.25
    
def initRand():
    return random.random()#*0.5

def multiGauss(n,size):
    z = torch.zeros(size=(size,size))
    for i in range(n+1):
        z_i   = singleGauss(size,AmpRand(),SigmaRand(),SigmaRand(),initRand(),initRand())
        z    += z_i
        z     = torch.functional.normalize(z)
    return z

def write_image(z,img_name):
    torchvision.utils.save_image(z,img_name)
