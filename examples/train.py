import os
import time
import copy
import torch
import glob
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
from gpatch_cnn.model import Gauss2D

show        = True
pretrain    = False
data_dir    = 'data'
filename    = 'model_out_gauss.pth'
batch_size  = 10       #Number of samples in each batch
num_workers = 4
epoch_num   = 10       #Number of epochs to train the network
lr          = 1e-5      # Learning rate


# calculate the number of batches per epoch
data_transforms = {
    'train': transforms.ToTensor(),
    'val': transforms.ToTensor()
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                                          for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
                                             for x in ['train', 'val']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

Gauss2D.imshow(out,title=[class_names[x] for x in classes], fname='initial.jpg', show=True)

model_ft = models.resnet18(pretrained=pretrain)
if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(),"gpus!")
    model_ft = nn.DataParallel(model_ft)

model_ft=model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

model_ft = Gauss2D.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=epoch_num, dataloaders=dataloaders, device=device,dataset_sizes=dataset_sizes)
torch.save(model_ft, filename)

Gauss2D.visualize_model(model_ft,num_images=10,device=device,dataloaders=dataloaders,class_names=class_names)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

nb_classes = 6

# Initialize the prediction and label lists(tensors)
predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)

        # Append batch prediction results
        predlist=torch.cat([predlist,preds.view(-1).cpu()])
        lbllist=torch.cat([lbllist,classes.view(-1).cpu()])

# Confusion matrix
conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
print(conf_mat)

# Per-class accuracy
class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
print(class_accuracy)
labels = ['0','1','2','3','4','5']
ax=plt.subplot()


sns.heatmap(conf_mat, annot=True)
class_names = np.asarray(labels)
ax.set_xlabel('Predicted # Peaks')
ax.set_ylabel('True # Peaks')
ax.set_xticklabels(class_names.flatten())
ax.set_yticklabels(class_names.flatten())
ax.set_title('Confusion Matrix')
plt.savefig('confusion_matrix_gauss_class.pdf')
