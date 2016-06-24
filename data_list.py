#encoding=utf8
import os, sys
import random
import cfgs
import warnings

train_ratio=.9

folders = [os.path.join(cfgs.data_path,cfgs.train_data_path)]
if os.path.exists(cfgs.data_list_path):
    pass
else:
    os.system("mkdir "+cfgs.data_list_path)
fid_train = open(os.path.join(cfgs.data_list_path,'train.txt'), 'w')
fid_val = open(os.path.join(cfgs.data_list_path,'val.txt'), 'w')

all_list = []
train_list = []
val_list = []
random.seed(0)


for folder in folders:
    print folder
    for img in os.listdir(folder):
        if len(img) == 0:
            raise ValueError("invalid name")
        if len(img.replace(' ','')) != len(img):
            warnings.warn("whitespace in name")
        filename = os.path.join(folder, img)
        all_list.append(filename) 


random.shuffle(all_list)
data_len=len(all_list)
print 'length of name list',data_len 
train_list=all_list[: int(data_len*train_ratio)]
print 'len_train_list', len(train_list)
val_list=all_list[int(data_len*train_ratio) :]
print 'len_val_list', len(val_list)


for line in train_list:
    fid_train.write(line + '\n')
fid_train.close()

for line in val_list:
    fid_val.write(line + '\n')
fid_val.close()
