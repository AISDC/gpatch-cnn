from gpatch_cnn.torch_gauss import genData 

train_split = 0.9 
num_images  = 2000
train_img   = int(num_images * train_split) 
val_img     = int(num_images * (1.0 - train_split))

train=genData(patch_size,train_img,n_peaks)
val=genData(patch_size,val_img,n_peaks)
