# by yihui
import caffe
from utils import factory
import cfgs_res as cfgs

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
    n_filter=32
    n.Data("/home/yihuihe/Ultrasound-Nerve-Segmentation/data/train.txt",backend='image',label="aaa",new_height=h,new_width=w,scale=0.01578412369702059, batch_size=batch_size)  

    n.Deconvolution('up1',)

    n.Data("data/mask.txt",name='label',mean_file="data/mask_mean.binaryproto",backend='image',new_height=h,new_width=w,batch_size=batch_size, scale=1./255)
    n.silence('nothing','aaa')
    n.diceLoss('prob','label')

    n.totxt('resnet/addition.prototxt')

    