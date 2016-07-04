# by yihui
import caffe
from utils import factory

def trainval():
    pass

def deploy():
    pass

def net():
    pass

if __name__ == '__main__':
    n=factory('unet')
    h=96
    w=128
    batch_size=32
    omit=0.3
    n.Data("/home/yihuihe/Ultrasound-Nerve-Segmentation/data/train.txt",backend='image',label="aaa",new_height=h,new_width=w,scale=0.01578412369702059, batch_size=batch_size) 
    n_filter=32
    for i in range(1,5):
        n.conv_relu(i,i,n_filter)
        n.conv_relu(i*10+1,i*10+1,n_filter)
        if omit != 0:
            n.Dropout(i,omit=omit)
        n.Pooling(i)
        n_filter*=2
        
    i=5
    n.conv_relu(i,i,n_filter)
    n.conv_relu(i*10+1,i*10+1,i*n_filter)
    n_filter/=2

    for i in range(6,10):
        n.Deconvolution(i, n_filter)
        n.Concat(i,bottom1=n.bottom,bottom2='conv'+str((10-i)*10+1))
        n.conv_relu(i,i,n_filter)
        n.conv_relu(i*10+1,i*10+1,n_filter)
        if omit!=0:
            n.Dropout(i,omit=omit)
        n_filter/=2

    n.Convolution(10,num_output=1,kernel_size=1,pad=0)
    n.Sigmoid()

    n.Data("data/mask.txt",name='label',backend='image',new_height=h,new_width=w,batch_size=batch_size, scale=1./255)
    n.silence('nothing','aaa')
    n.diceLoss('prob','label')
    

    n.totxt('unet/trainval.prototxt')
    n.totxt('unet/test.prototxt')




