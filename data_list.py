#encoding=utf8
import os, sys
import random
import cfgs
import warnings
import cv2
import numpy as np
import pandas as pd
train_ratio=.9

folders = [cfgs.train_data_path]
if os.path.exists(cfgs.data_list_path):
    pass
else:
    os.system("mkdir "+cfgs.data_list_path)

all_list = []
train_list = []
val_list = []
mask_list = []


for folder in folders:
    print folder
    for img in os.listdir(folder):
        if len(img) == 0:
            raise ValueError("invalid name")
        if len(img.replace(' ','')) != len(img):
            warnings.warn("whitespace in name")
        filename = os.path.join(folder, img)
        img=cv2.imread(filename)
        # print np.histogram(img)
        mask_filename=filename.replace('train_data','train_mask').replace('.tif','_mask.tif')
        img=cv2.imread(mask_filename)

        mask_list.append(mask_filename)
        if np.any(img>50):
            all_list.append([filename,1])
        else:
            all_list.append([filename,0])


random.seed(1234)
random.shuffle(all_list)
random.seed(1234)
random.shuffle(mask_list)

data_len=len(all_list)
print 'length of name list',data_len 
train_list=all_list[: int(data_len*train_ratio)]
train_mask_list=mask_list[: int(data_len*train_ratio)]
print 'len_train_list', len(train_list)
val_list=all_list[int(data_len*train_ratio) :]
val_mask_list=mask_list[int(data_len*train_ratio) :]
print 'len_val_list', len(val_list)



with open(os.path.join(cfgs.data_list_path,'train.txt'), 'w') as f:
    for line in train_list:
        f.write(line[0]+' '+ str(line[1]) + '\n')

with open(cfgs.data_list_path+'/mask.txt','w') as f:
    for line in train_mask_list:
        f.write(line + ' 0\n')

with open(cfgs.data_list_path+'/val_mask.txt','w') as f:
    for line in val_mask_list:
        f.write(line + ' 0\n')

with open(os.path.join(cfgs.data_list_path,'val.txt'), 'w') as f:
    for line in  val_list:
        f.write(line[0]+' '+ str(line[1]) + '\n')
