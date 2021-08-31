#!/usr/bin/env python

import os
import torch
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colors
#from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans 
from sklearn.metrics import calinski_harabasz_score
from scipy.interpolate import UnivariateSpline
from torchvision import datasets, models, transforms, utils
from sklearn.preprocessing import StandardScaler

data_dir        = 'sensor_frames'
num_workers     = 8
#crop_size       = 8
data_transforms = transforms.Compose([transforms.Grayscale(),
                                      transforms.ToTensor()])
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


image_dataset   = datasets.ImageFolder(os.path.join(data_dir),data_transforms)
dataloader      = torch.utils.data.DataLoader(image_dataset,num_workers=num_workers,batch_size=len(image_dataset))
images,labels   = next(iter(dataloader))
images = images.to(device, torch.uint8)


images = images.reshape(len(images),-1)
#images = images.cpu().numpy()
#images = StandardScaler().fit_transform(images)


clusters=range(2,20)

summed_square_distance=[]
calinski_score=[]

for i in clusters:
    cluster_ids,cluster_centers=kmeans(X=images,num_clusters=i,distance='euclidean',device=torch.device('cuda'))
    calinski_score.append(calinski_harabasz_score(images,cluster_ids))

idx_max = max(range(len(calinski_score)),key=calinski_score.__getitem__)
n_clusters=idx_max + min(clusters)


plt.figure()
plt.plot(clusters,spline_d2(clusters))
plt.show()


print(n_clusters)
labels,centers = kmeans(X=images,num_clusters=n_clusters,distance='euclidean',device=torch.device('cuda'))
Y = labels


#PCA to 2D
pca           = PCA(n_components=2)
pca_transform = pca.fit_transform(images)

plt.figure()
fig=plt.figure()
ax=fig.add_subplot(111)
i=0
pca_holder = pd.DataFrame(columns=['pca0','pca1','label'])
pca_holder['pca0']   = pca_transform[:,0]
pca_holder['pca1']   = pca_transform[:,1]
pca_holder['labels'] = Y
unique_labels = set(pca_holder['labels'])
plt.scatter(pca_transform[:,0],pca_transform[:,1], c=Y,cmap=plt.cm.jet)
plt.tight_layout()
plt.savefig('pca_plot.png')

def infer_cluster_labels(n_clusters, km_labels, actual_labels):
    inferred_labels = {}
    for i in range(n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(km_labels == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

        #print(labels)
        #print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))

    return inferred_labels


centroids = centers
images = centroids.reshape(n_clusters, crop_size, crop_size)
images *= 255
images = images.astype(np.uint8)

# determine cluster labels
cluster_labels = infer_cluster_labels(n_clusters,labels,Y)



# create figure with subplots using matplotlib.pyplot
fig, axs = plt.subplots(1, 5, figsize = (20, 20))
plt.gray()

# loop through subplots and add centroid images
for i, ax in enumerate(axs.flat):

    # determine inferred label using cluster_labels dictionary
    for key, value in cluster_labels.items():
        if i in value:
            ax.set_title('Inferred Label: {}'.format(key))

    # add image to subplot
    ax.matshow(images[i])
    ax.axis('off')

# display the figure
fig.show()
