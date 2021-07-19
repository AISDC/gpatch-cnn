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

def imshow(inp, title=None, fname=None, show=True):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if show is True: 
        plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    if fname is not None: 
        plt.savefig('%s' %fname)

## test //train model 
def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, device, dataset_sizes):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for ep in range(num_epochs):  # epochs loop
       print('Epoch {}/{}'.format(ep, num_epochs - 1))
       print('-' * 10)
    
       for phase in ['train', 'val']:
           if phase == 'train':
               scheduler.step()
               model.train()  # Set model to training mode
           else:
               model.eval()   # Set model to evaluate mode
    
           running_loss = 0.0
           running_corrects = 0
    
           # Iterate over data.
           for inputs, labels in dataloaders[phase]:
               inputs = inputs.to(device)
               labels = labels.to(device)
    
               # zero the parameter gradients
               optimizer.zero_grad()
    
               # forward
               # track history if only in train
               with torch.set_grad_enabled(phase == 'train'):
                   outputs = model(inputs)
                   _, preds = torch.max(outputs, 1)
                   loss = criterion(outputs, labels)
    
                   # backward + optimize only if in training phase
                   if phase == 'train':
                       loss.backward()
                       optimizer.step()
    
               # statistics
               running_loss += loss.item() * inputs.size(0)
               running_corrects += torch.sum(preds == labels.data)
    
           epoch_loss = running_loss / dataset_sizes[phase]
           epoch_acc = running_corrects.double() / dataset_sizes[phase]
    
           print('{} Loss: {:.4f} Acc: {:.4f}'.format(
               phase, epoch_loss, epoch_acc))
    
           # deep copy the model
           if phase == 'val' and epoch_acc > best_acc:
               best_acc = epoch_acc
               best_model_wts = copy.deepcopy(model.state_dict())
      
       print()
       time_elapsed = time.time() - since
       print('Training complete in {:.0f}m {:.0f}s'.format(
           time_elapsed // 60, time_elapsed % 60))
       print('Best val Acc: {:4f}'.format(best_acc))
    
       # load best model weights
       model.load_state_dict(best_model_wts)
       #return model
       return model;     

def visualize_model(model, dataloaders, device, class_names, num_images):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print('actual labels')

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                #print(class_names[labels[j]]) 
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                fig.tight_layout() 
                ax.axis('off')
                ax.set_title('predicted: {}\nactual: {}' .format(class_names[preds[j]], class_names[labels[j]]))
                imshow(inputs.cpu().data[j],fname='model_predict.jpg')


                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

