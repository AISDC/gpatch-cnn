from gpatch_cnn.t_gauss import genTrainData, genValData

train_split = 0.9 
num_images  = 2000
train_img   = int(num_images * train_split) 
val_img     = int(num_images * (1.0 - train_split))

genTrainData(train_img)

genValData(val_img)

