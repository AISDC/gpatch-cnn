import os
import torch
import random
import pandas as pd
import numpy as np
import torchvision
#from torchvision.ops import box_iou
#from skimage.measure import label, regionprops

def singleGauss(size,amp,sigma_x,sigma_y,x0,y0):
    x    = torch.linspace(0, 1, size)
    y    = torch.linspace(0, 1, size)
    x, y = torch.meshgrid(x, y)
    z    = (1/(2*np.pi*sigma_x*sigma_y) * amp*np.exp(-((x-x0)**2/(2*sigma_x**2)
       		+ (y-y0)**2/(2*sigma_y**2))))
    return z

def AmpRand():
    return 0.1#random.random()

def SigmaRand():
    return random.random()/8

def initRand():
    return random.random()

def distance(x1,y1,x2,y2):
    d_square = (x1-x2)**2 + (y1-y2)**2
    return np.sqrt(d_square)

def fwhm(sigma): 
    return 2*np.sqrt(2*np.log(2)) * sigma

def multiGauss(n,size):
    min_i = 0
    max_i = size
    z = torch.zeros(size=(size,size))
    for i in range(n+1):
        z_i   = singleGauss(size,AmpRand(),SigmaRand(),SigmaRand(),initRand(min_i,max_i),initRand())
        z    += z_i
    return z

def export_positions(coord_list,n,idx):
    os.makedirs('positions/pos_%i' %n,exist_ok=True)
    df = pd.DataFrame(coord_list,columns=['x','y'])
    df.to_csv('positions/pos_%i/peaks_img_%i.csv' %(n,idx))

def export_pos(coord_list,n,dir_loc):
    os.makedirs(dir_loc,exist_ok=True)
    df = pd.DataFrame(coord_list,columns=['x','y'])
    df.to_csv(dir_loc+'/data_%i.csv' %n)

def multiGaussNoOverlap(n,size,cutoff,idx):
    tot_cords  = []
    coord_list = []
    if n > 0:
        #initialize list w/first entry
        x_i,y_i  = (initRand(),initRand())
        coord_list.append((x_i,y_i))
    while len(coord_list) <  n:
        #create test x,y
        x_j,y_j  = (initRand(),initRand())
        sigma_x    = SigmaRand() 
        sigma_y    = SigmaRand() 
        max_fwhm   = max([fwhm(sigma_x),fwhm(sigma_y)]) 
        #check that new point is far enough
        #from others in list
        for coord in coord_list:
            x_n,y_n = coord
            #if so, add to list
#            if distance(x_n,y_n,x_j,y_j) > cutoff:
            if distance(x_n,y_n,x_j,y_j) > 2*max_fwhm:
                coord_list.append((x_j,y_j))
                break
            else:
                break
    z = torch.zeros(size=(size,size))
    for coord in coord_list:
        x_i,y_i = coord
        z_i   = singleGauss(size,AmpRand(),SigmaRand(),SigmaRand(),x_i,y_i)
        z    += z_i
    return z,coord_list

def write_image(z,img_name):
    torchvision.utils.save_image(z,img_name)
