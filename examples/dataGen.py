import os 
from gpatch_cnn.torch_gauss import multiGauss, write_image 

patch_size  = 16#128
n_peaks     = 5
train_n     = 1800
val_n       = 200


for n in range(1,n_peaks+1):
    train_path = 'data/train/%i' %n 
    os.makedirs(train_path,exist_ok=True)
    for i in range(train_n):
        img = multiGauss(n,patch_size)
        write_image(img,train_path+'/img_%i.png' %i)


for n in range(1,n_peaks+1): 
    val_path = 'data/val/%i' %n  
    os.makedirs(val_path,exist_ok=True)
    for j in range(val_n): 
        img = multiGauss(n,patch_size)
        write_image(img,val_path+'/img_%i.png' %i)
