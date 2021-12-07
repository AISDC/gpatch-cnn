import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

# pytorch relates imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

#model params
data_dir    = 'data/train'
num_workers = 8


#model parameters
batch_size    = 1
num_epochs    = 100
learning_rate = 1e-3
size_in       = 1024 
size_h1       = 100
size_h2       = 10
size_out      = 2 
#size_h4       = 1

data_transforms = transforms.Compose([transforms.Grayscale(),
                                      transforms.ToTensor()])
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_dataset   = datasets.ImageFolder(os.path.join(data_dir),data_transforms)
dataloader      = torch.utils.data.DataLoader(image_dataset,num_workers=num_workers,batch_size=len(image_dataset))
images,labels   = next(iter(dataloader))
print("number of images in dataset: %i" %(len(image_dataset)))

images = images.to(device, torch.uint8)
images = images.reshape(len(images),-1)
positions = pd.read_csv('positions/train/data.csv') 

X = images
Y = torch.Tensor(np.array((positions['x'],positions['y'])).reshape(len(positions),2))#,positions_train['y'])

print(X.size(),Y.size())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

criterion = nn.MSELoss()

class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(size_in, size_h1)
        self.lin2 = nn.Linear(size_h1, size_h2)
        self.lin3 = nn.Linear(size_h2, size_out)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, input):
        x = self.relu1(self.lin1(input))
        x = self.relu2(self.lin2(x))
        x = self.lin3(x)
        return x

net=TorchModel()

def train(model_inp, device, num_epochs = num_epochs):
    optimizer = torch.optim.RMSprop(model_inp.parameters(),lr=learning_rate)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_acc = 0.0
        for inputs, labels in train_iter:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model_inp(inputs)
            # defining loss
            loss = criterion(outputs, labels)

            # computing gradients
            loss.backward()
            # updated weights based on computed gradients
            optimizer.step()
            # accumulating running loss
            running_loss += loss.item()
   
        print('Epoch [%d]/[%d] running loss: %.6f' %
                  (epoch + 1, num_epochs, running_loss/len(train_iter)))
        running_loss = 0.0

def train_load_save_model(model_obj, model_path, device):
    # train model
    train(model_obj,device)
    print('Finished training the model. Saving the model to the path: {}'.format(model_path))
    torch.save(model_obj, model_path)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=2)

X_train = X_train.float()
y_train = y_train.float() #.view(-1, 1).float()

X_test = X_test.float()
y_test = y_test.float()#.view(-1, 1).float()


print(X_train.size(),y_train.size())
datasets = torch.utils.data.TensorDataset(X_train, y_train)
train_iter = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=True)


test_data = torch.utils.data.TensorDataset(X_test)
test_iter = torch.utils.data.DataLoader(test_data, batch_size=1)

model = TorchModel()
if torch.cuda.device_count() > 1: 
    print("Using ", torch.cuda.device_count(),"gpus!")
    model = nn.parallel.DataParallel(model)

model = model.to(device)
model.train()

path = 'models/spectral_model.pt'
train_load_save_model(model,path,device)

from sklearn.metrics import mean_squared_error

model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    y_test_pred = model(X_test)
    y_test_pred.cpu().numpy()

    X_train = X_train.to(device)
    y_pred = model(X_train)
    y_pred.cpu().numpy()

train_err = np.sqrt(mean_squared_error(y_pred.cpu().numpy(), Y.cpu().numpy()))
test_err = np.sqrt(mean_squared_error(y_test_pred.cpu().numpy(), y_test.cpu().numpy()))

print('Train MSE err: ', train_err)
print('Test MSE err: ', test_err) 
