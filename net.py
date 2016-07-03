# by yihui
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop
from utils import factory

def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='msra'))
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def fcn(train,mask,batch_size=8):
    n = caffe.NetSpec()
    # n.data, n.sem, n.geo = L.Python(module='siftflow_layers',
    #         layer='SIFTFlowSegDataLayer', ntop=3,
    #         param_str=str(dict(siftflow_dir='../data/sift-flow',
    #             split=split, seed=1337)))

    n.data =L.Data(backend=P.Data.LMDB,batch_size=batch_size, source=train,
                             transform_param=dict(scale=1./255),ntop=1
                  )

    n.geo = L.Data(backend=P.Data.LMDB, batch_size=batch_size, source=mask,
                         ntop=1)
    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 32)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 32)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 64)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 64)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 128)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 128)
    n.pool3 = max_pool(n.relu3_2)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 256)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 256)
    n.pool4 = max_pool(n.relu4_2)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)

    n.up6=L.Concat(L.Interp(n.relu5_2),n.conv5_2)
    

    n.loss_geo = L.SoftmaxWithLoss(n.up6, n.geo,
            loss_param=dict(normalize=False))#, ignore_label=255))

    return n.to_proto()

def make_net():
    with open('trainval.prototxt', 'w') as f:
        f.write(str(fcn('/mnt/data1/yihuihe/selected_data/gen_shantian12_pos_and_neg/lmdb_train_val/train_data','/mnt/data1/yihuihe/selected_data/gen_shantian12_pos_and_neg/lmdb_train_val/train_mask')))

    with open('test.prototxt', 'w') as f:
        f.write(str(fcn('/mnt/data1/yihuihe/selected_data/gen_shantian12_pos_and_neg/lmdb_train_val/val_data','/mnt/data1/yihuihe/selected_data/gen_shantian12_pos_and_neg/lmdb_train_val/val_mask',1)))

if __name__ == '__main__':
    n=factory('unet')
    
    n.Data("/mnt/data1/yihuihe/ultrasound-nerve/lmdb_train_val/train_data",mean_file="data/data_mean.binaryproto",scale=0.01578412369702059)
    n_filter=32
    for i in range(1,5):
        n.conv_relu(i,i,n_filter)
        n.conv_relu(i*10+1,i*10+1,n_filter)
        n.Pooling(i)
        n_filter*=2
        
    i=5
    n.conv_relu(i,i,n_filter)
    n.conv_relu(i*10+1,i*10+1,i*n_filter)
    n_filter/=2

    for i in range(6,10):
        n.Deconvolution(i, n_filter, bottom='relu'+str((i-1)*10+1))
        n.Concat(i,bottom1=n.bottom,bottom2='conv'+str(10-i))
        n.conv_relu(i,i,n_filter)
        n.conv_relu(i*10+1,i*10+1,n_filter)
        n_filter/=2

    n.Convolution(10,num_output=1,kernel_size=1,pad=0)
    n.Sigmoid()

    n.Data("/mnt/data1/yihuihe/ultrasound-nerve/lmdb_train_val/train_data",name='label',mean_file="data/mask_mean.binaryproto")
    n.diceLoss('prob','label')
    

    n.totxt('unet/trainval.prototxt')




