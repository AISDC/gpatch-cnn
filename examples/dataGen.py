import os 
from gpatch_cnn.torch_gauss import multiGauss, multiGaussNoOverlap, write_image, export_pos 

patch_size  = 32
n_peaks     = 1
train_n     = 10000
val_n       = 2000
cutoff      = 0.4

train_pos=[]
for n in range(1,n_peaks+1):
    train_path    = 'data/train/%i' %n
    train_pos_dir = 'positions/train'
    os.makedirs(train_path,exist_ok=True)
    for i in range(train_n):
        img,pos = multiGaussNoOverlap(n,patch_size,cutoff,i)
        write_image(img,train_path+'/img_%i.png' %i)
        train_pos.extend(pos)
    export_pos(train_pos,n,train_pos_dir)

test_pos=[]
for n in range(1,n_peaks+1): 
    val_path     = 'data/val/%i' %n
    test_pos_dir = 'positions/val'
    os.makedirs(val_path,exist_ok=True)
    for j in range(val_n): 
        img,pos = multiGaussNoOverlap(n,patch_size,cutoff,i)
        write_image(img,val_path+'/img_%i.png' %j)
        test_pos.extend(pos)
    export_pos(test_pos,n,test_pos_dir)
