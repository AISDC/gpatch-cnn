#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable
from model import Gauss2D 

data_dir = 'data/val'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    model=torch.load('model/model_out_gauss.pth',map_location=torch.device('cpu'))
else: 
    model=torch.load('model/model_out_gauss.pth')
model.eval()

to_pil = transforms.ToPILImage()
images, labels, classes = Gauss2D.get_random_images(5,data_dir=data_dir)
fig=plt.figure(figsize=(10,10))
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = Gauss2D.predict(image,model,device,labels)
    sub = fig.add_subplot(1, len(images), ii+1)
    res = int(labels[ii]) == index
    sub.set_title(str(classes[index]) + ":" + str(res))
    plt.axis('off')
    plt.imshow(image)
plt.savefig('model_inference.png')

