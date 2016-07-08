# by yihui
import caffe
from utils import factory
import unet_cfgs as cfgs

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
    batch_size=32
    omit=0
    layers=5
    if use_lmdb:
        n.Data("/mnt/data1/yihuihe/ultrasound-nerve/lmdb_train_val/train_data",scale=0.01578412369702059,backend='LMDB')
    else:
        n.Data("data/mask.txt",name='label',backend='image',new_height=h,new_width=w,batch_size=batch_size, scale=1./255)
        n.Data("data/train.txt",backend='image',label="aaa",new_height=h,new_width=w,scale=1./255,batch_size=batch_size) 
    n_filter=32
    for i in range(1,layers):
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
        n.Deconvolution(i, n_filter)# 2*n_filter)
        # n.Convolution([i,0],n_filter)
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
        n.Data("/mnt/data1/yihuihe/ultrasound-nerve/lmdb_train_val/train_data",name='label',scale=1./255)
    else:
        n.silence('nothing','aaa')
    n.diceLoss('prob','label')
    

    n.totxt('unet/trainval.prototxt')
    n.totxt('unet/test.prototxt')




