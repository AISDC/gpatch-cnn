import os 
from gpatch_cnn.torch_gauss import multiGauss, multiGaussNoOverlap, write_image 

patch_size  = 256
n_peaks     = 5
train_n     = 10
val_n       = 2
cutoff      = 0.2

for n in range(n_peaks):
    train_path = 'data/train/%i' %n 
    os.makedirs(train_path,exist_ok=True)
    for i in range(train_n):
        img = multiGaussNoOverlap(n,patch_size,cutoff)
        write_image(img,train_path+'/img_%i.png' %i)


#for n in range(n_peaks): 
#    val_path = 'data/val/%i' %n  
#    os.makedirs(val_path,exist_ok=True)
#    for j in range(val_n): 
#        img = multiGaussNoOverlap(n,patch_size,cutoff)
#        write_image(img,val_path+'/img_%i.png' %i)
