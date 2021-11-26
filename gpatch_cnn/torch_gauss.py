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
    return random.random()*0.5
    
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
        z     = torch.nn.functional.normalize(z)
    return z

def multiGaussNoOverlap(n,size,cutoff):
    coord_list = []
    if n > 0: 
        #initialize list w/first entry
        x_i,y_i  = (initRand(),initRand()) 
        coord_list.append((x_i,y_i))
    while len(coord_list) <  n:
        #create test x,y              
        x_j,y_j  = (initRand(),initRand())
        #check that new point is far enough 
        #from others in list 
        for coord in coord_list:
            x_n,y_n = coord
            #if so, add to list
            if distance(x_n,y_n,x_j,y_j) > cutoff:
                coord_list.append((x_j,y_j))
                break
            else: 
                break
    z = torch.zeros(size=(size,size))
    for coord in coord_list: 
        x_i,y_i = coord
        z_i   = singleGauss(size,AmpRand(),SigmaRand(),SigmaRand(),x_i,y_i)
        z    += z_i
#        z     = torch.nn.functional.normalize(z)
    return z


def write_image(z,img_name):
    torchvision.utils.save_image(z,img_name)
