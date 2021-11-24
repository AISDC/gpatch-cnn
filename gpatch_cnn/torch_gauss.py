import torch
import random
import numpy as np
import torchvision
from torchvision.ops import box_iou
from skimage.measure import label, regionprops

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
    return random.random()

def distance(x1,y1,x2,y2):
    d_square = (x1-x2)**2 + (y1-y2)**2
    return np.sqrt(d_square)

def multiGauss(n,size):
    z = torch.zeros(size=(size,size))
    for i in range(n+1):
        z_i   = singleGauss(size,AmpRand(),SigmaRand(),SigmaRand(),initRand(),initRand())
        z    += z_i
#        z     = torch.nn.functional.normalize(z)
    return z

def multiGaussNoOverlap(n,size,cutoff):
    coord_list = []
    x_i = initRand()
    y_i = initRand()
    z = torch.zeros(size=(size,size))
    i=0
    while i < n:
        coord_list = [] 
        coord_init = (initRand(),initRand())        
        coord_list.append(coord_init)              
        x_i   = initRand() 
        y_i   = initRand()
        for coord in coord_list:
            x_j,y_j = coord 
            if distance(x_i,y_i,x_j,y_j) > cutoff:
                coord_list.append((x_i,y_i))
                i +=1
    for coord in coord_list: 
        x_i,y_i = coord
        z_i   = singleGauss(size,AmpRand(),SigmaRand(),SigmaRand(),x_i,y_i)
        z    += z_i
        z     = torch.nn.functional.normalize(z)
    return z


def write_image(z,img_name):
    torchvision.utils.save_image(z,img_name)
