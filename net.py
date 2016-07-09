# by yihui
import caffe
from utils import factory
import unet_cfgs as cfgs
import numpy as np


use_lmdb=False
def trainval():
    pass

def deploy():
    pass

def net():
    pass

if __name__ == '__main__':
    n=factory('unet')
    h=cfgs.inShape[0]
    w=cfgs.inShape[1]
    batch_size=24
    omit=0
    layers=5
    if use_lmdb:
        n.Data("/mnt/data1/yihuihe/ultrasound-nerve/lmdb_train_val/train_data", #mean_file='data/data_mean.binaryproto',
        mean_value=[128],
        scale=1./255,
        backend='LMDB')
    else:
        n.Data("data/val_mask.txt",
            name='label',
            backend='image',
            new_height=h,
            new_width=w,
            batch_size=batch_size, 
            scale=1./255,
            phase="TEST"
            )
        n.Data("data/val.txt",
            backend='image',
            label="aaa",
            new_height=h,
            new_width=w,
            batch_size=batch_size, 
            scale=1./255,
            phase="TEST"
            ) 
        n.Data("data/mask.txt",
            name='label',
            backend='image',
            new_height=h,
            new_width=w,
            batch_size=batch_size, 
            scale=1./255
            )
        n.Data("data/train.txt",
            backend='image',
            label="aaa",
            new_height=h,
            new_width=w,
            batch_size=batch_size, 
            scale=1./255
            )
        n.Python("preprocessing",'transformLayer', bottom=['data','label','aaa','nothing'], top=['newdata','newlabel','hasObj'])
    n_filter=32
    for i in range(1,layers):
        if i ==1:
            n.conv_relu(i,i,n_filter,bottom='newdata')
        else:
            n.conv_relu(i,i,n_filter)
        n.conv_relu([i,1],[i,1],n_filter)
        if omit != 0:
            n.Dropout(i,omit=omit)
        n.Pooling(i)
        n_filter*=2
        
    i=layers
    n.conv_relu(i,i,n_filter)
    n.conv_relu([i,1],[i,1],n_filter)
    n_filter/=2

    for i in range(layers+1,layers*2):
        n.Deconvolution(i, 2*n_filter)
        n.Convolution([i,0],n_filter)
        low_level='conv_'+str(2*layers-i)+'_1'
        # low_level='conv'+str(2*layers-i)
        n.Concat(i,bottom1=n.bottom,bottom2=low_level)
        n.conv_relu(i,i,n_filter)
        n.conv_relu([i,1],[i,1],n_filter)
        if omit!=0:
            n.Dropout(i,omit=omit)
        n_filter/=2

    n.Convolution(layers*2,num_output=1,kernel_size=1,pad=0)
    n.Sigmoid()

    if use_lmdb:
        n.Data("/mnt/data1/yihuihe/ultrasound-nerve/lmdb_train_val/train_mask",name='label', backend='LMDB')
    else:
        pass
        # n.silence('nothing')
    n.diceLoss('prob','newlabel')
    
    # has object prediction
    n.conv_relu('conv_classifier', "classifier_relu",1024,stride=2, bottom= 'conv_5_1')
    n.Pooling('global_pooling',pool="AVE", global_pooling=True)
    n.fc_relu_dropout('FClayer','fc_relu','fc_dropout',256)
    n.InnerProduct('bool_classifier', 2)
    n.SoftmaxWithLoss(loss_weight=0, bottom1='bool_classifier', bottom2='hasObj')
    

    n.totxt('unet/trainval.prototxt')
    # n.totxt('unet/test.prototxt')




