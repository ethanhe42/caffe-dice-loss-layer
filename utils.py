#-.-encoding=utf-8-.-``
# Yihui He, https://yihui-he.github.io
''' My caffe helper'''
import caffe
from caffe import layers as L
import os
import warnings
import cv2
import numpy as np
import pandas as pd
import PIL.Image as Image
import sys
import lmdb
import random
from easydict import EasyDict as edict

class CaffeSolver:
    
    """
    Caffesolver is a class for creating a solver.prototxt file. It sets default
    values and can export a solver parameter file.
    Note that all parameters are stored as strings. Strings variables are
    stored as strings in strings.
    """

    def __init__(self, debug=False):

        self.sp = {}
        self.sp['test_net']="testnet.prototxt"
        self.sp['train_net']="trainnet.prototxt"
        
        # critical:
        self.sp['base_lr'] = 0.001
        self.sp['momentum'] = 0.9

        # speed:
        self.sp['test_iter'] = 100
        self.sp['test_interval'] = 250

        # looks:
        self.sp['display'] = 25
        self.sp['snapshot'] = 2500
        self.sp['snapshot_prefix'] = 'snapshot'  # string withing a string!

        # learning rate policy
        self.sp['lr_policy'] = 'fixed' # see caffe proto
#   //    - fixed: always return base_lr.
#   //    - step: return base_lr * gamma ^ (floor(iter / step))
#   //    - exp: return base_lr * gamma ^ iter
#   //    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
#   //    - multistep: similar to step but it allows non uniform steps defined by
#   //      stepvalue
#   //    - poly: the effective learning rate follows a polynomial decay, to be
#   //      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
#   //    - sigmoid: the effective learning rate follows a sigmod decay
#   //      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
        
        self.sp['gamma'] = 1   

        self.sp['weight_decay'] = 0.0005

        # pretty much never change these.
        self.sp['max_iter'] = 100000
        self.sp['test_initialization'] = False
        self.sp['average_loss'] = 25  # this has to do with the display.
        self.sp['iter_size'] = 1  # this is for accumulating gradients

        if (debug):
            self.sp['max_iter'] = 12
            self.sp['test_iter'] = 1
            self.sp['test_interval'] = 4
            self.sp['display'] = 1

    def add_from_file_notavailablenow(self, filepath):
        """
        Reads a caffe solver prototxt file and updates the Caffesolver
        instance parameters.
        """
        with open(filepath, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                splitLine = line.split(':')
                self.sp[splitLine[0].strip()] = splitLine[1].strip()

    def write(self, filepath):
        """
        Export solver parameters to INPUT "filepath". Sorted alphabetically.
        """
        f = open(filepath, 'w')
        for key, value in sorted(self.param2str(self.sp).items()):
            if not(type(value) is str):
                raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))
    
    def param2str(self,sp):
        for i in sp:
            if isinstance(sp[i],str):
                sp[i]='"'+sp[i]+'"'
            elif isinstance(sp[i],bool):
                if sp[i]==True:
                    sp[i]='true'
                else:
                    sp[i]='false'
            else:
                sp[i]=str(sp[i])
        return sp

class Data:
    """Helper for dealing with DIY data"""
    def __init__(self):
        pass

    @staticmethod
    def folder_opt(folder, func, *args):
        """A func operate on each image then return a list"""
        all_list=[]
        for img in os.listdir(folder):
            if len(img) == 0:
                raise ValueError("invalid name")
            if len(img.replace(' ','')) != len(img):
                warnings.warn("whitespace in name")
            filename = os.path.join(folder, img)
            ret=func(filename, *args)
            if ret is None:
                continue
            all_list.append(ret)
        return all_list

    @staticmethod
    def imFromFile(path,dtype=np.float32):
        """load image as array from path"""
        return np.array(Image.open(path),dtype=dtype)

    @classmethod
    def showIm(cls,im,wait=-1,name='image'):
        """show arr image or image from directory and wait"""
        if isinstance(im,str):
            im=cls.imFromFile(im)
        if im.max()>1:
            im=im/255.0
        cv2.imshow(name,im)
        cv2.waitKey(wait)
    
    @classmethod
    def saveIm(cls,im,path=''):
        """save im to path"""
        pass

    @classmethod
    def splitPath(cls,file):

        base=os.path.basename(file)
        dir=os.path.dirname(file)
        name,ext=os.path.splitext(base)
        return dir,name,ext
    
    @classmethod
    def splitDataset(cls,folder,train_ratio=.9, save_dir=None, seed=0):
        """split DIY data in a folder into train and val"""
        if isinstance(folder, str):
            folders=[folder]
        else:
            folders=folder
        if save_dir is None:
            save_dir=os.path.join(os.path.dirname(folder),os.path.basename(folder)+"_list")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        all_list = []
        train_list = []
        val_list = []
        if seed is not None:
            random.seed(seed)

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

        fid_train = open(os.path.join(save_dir,'train.txt'), 'w')
        fid_val = open(os.path.join(save_dir,'val.txt'), 'w')
        for line in train_list:
            fid_train.write(line + '\n')
        fid_train.close()

        for line in val_list:
            fid_val.write(line + '\n')
        fid_val.close()

        @classmethod
        def im2lmdb():
            """lmdb from DIY images"""
            pass


class NetHelper:
    """Helper for dealing with net"""
    def __init__(self, net=None,deploy=None,model=None):
        if net is not None:
            self.net=net
        else:
            self.netFromFile(deploy,model)

    def netFromFile(self,deploy_file,model_file,mode=caffe.TEST):
        self.net=caffe.Net(deploy_file,model_file,mode)
    

    @classmethod
    def gpu(cls,id=0):
        """open GPU"""
        caffe.set_device(id)
        caffe.set_mode_gpu()
        
    def prediction(self,c_img):
        """make prediction on single img"""
        if len(c_img.shape)==3:
            # color img
            depth=c_img.shape[2]
            pass
        elif len(c_img.shape)==2:
            # grey img
            tmp = np.zeros(c_img[:,:,np.newaxis].shape)
            tmp[:,:,0]=c_img
            c_img=tmp
            depth=1
        else:
            raise ValueError("abnormal image")
        c_img  = c_img.swapaxes(1,2).swapaxes(0,1) 
        # in Channel x Height x Width order (switch from H x W x C)
        c_img  = c_img.reshape((1,depth,c_img.shape[1],c_img.shape[2]))
        # 1 means batch size one
        prediction = self.net.forward_all(**{self.net.inputs[0]: c_img})
        return prediction

    def bin_pred_map(self,c_img, last_layer='prob',prediction_map=0):
        """get binary probability map prediction"""
        pred=self.prediction(c_img)
        prob_map=np.single(pred[last_layer][0,prediction_map,:,:])
        return prob_map

    def value_counts(self,layer): 
        print pd.value_counts(self.net.blobs[layer].data.flatten())
        
    def hist(self,layer, filters=None, bins=4, attr="blobs"):
        """
        inspect network response
        Args:
            filters: True will draw hist of every depth
        """
        if attr=="params" or attr=="param":
            response=self.net.params[layer][0].data
        elif attr=="diff":
            response=self.net.blobs[layer].diff
        else:
            response=self.net.blobs[layer].data
        if filters is None:
            # show response of this layer together
            cnts,boundary = np.histogram(response.flatten(),bins=bins)
            ret=pd.DataFrame(cnts,index=boundary[1:],columns=[layer])
            print ret.T
        else:
            # print every filter
            response=response.swapaxes(0,1)
            for filter in range(np.minimum(filters, response.shape[0])):
                cnts, boundary = np.histogram(response[filter,:,:,:].flatten(),bins=bins)
                ret=pd.DataFrame(cnts,index=boundary[1:],columns=[layer])
                print ret.T
                

    
    def layerShape(self,layer):
        """inspect network params shape"""
        response=self.net.blobs[layer].data
        print response.shape
    
    def showFilter(self,layer,filter=0,wait=-1,name='image'):
        """imshow filter"""
        response=self.net.blobs[layer].data[0,filter,:,:]
        Data.showIm(response,wait=wait,name=name)

class segmentation:
    """package for image segmentation"""
    def __init__(self):
        self.dices=[]
        self.recall=[]
        self.precision=[]
        self.neg_recall=[]
        self.cnt=0
    
    def update(self,pred,label):
        """
        update evaluation info using new pred and label, they must be the same size
        """
        if pred.shape!=label.shape:
            raise ValueError("pred and label not the same shape")
        intersection=np.sum(pred & label)
        pixelSum=np.sum(pred)+np.sum(label)
        if pixelSum==0:
            Dice=1.0
        else:
            Dice=2.0*intersection/pixelSum
        self.dices.append(Dice)

        if label.sum()==0:
            self.recall.append(1.0)
        else:
            self.recall.append(intersection*1.0/label.sum())
        
        if pred.sum()==0:
            self.recall.append(1.0)
        else:
            self.recall.append(intersection*1.0/pred.sum())
            
        self.neg_recall.append(np.sum((~pred)&(~label))*1.0/(~label).sum())
        # bug: base on no all positive label
        self.cnt+=1

    def show(self):
        """show histogram and metrics"""
        self.__print(self.dices,"dice")
        self.__print(self.recall,"recall")
        self.__print(self.precision,"precision")
        self.__print(self.neg_recall,"neg recall")

    def __print(self,metric,name='metric'):
        """print metric and hist"""
        print name,np.array(metric).mean()
        print np.histogram(metric)

class factory:
    """Helper for building network"""        
    def __init__(self, name='network'):
        self.proto='name: "'+name+'"\n'
        self.bottom=name
        self.string_field=set(['name','type','bottom','top','source','mean_file','module','layer'])

    def Data(self, 
        source,
        mean_file=None,
        mean_value=[],
        name='data', 
        scale=1,
        batch_size=32,
        backend='LMDB',
        new_height=None,
        new_width=None,
        is_color=False,
        label="nothing",
        phase='TRAIN'):
        p=[('name',name)]
        if backend=="LMDB":
            p+=[('type','Data'),
                ('top',name)]
        elif backend=="image":
            p+=[('type',"ImageData"),
                ('top',name),
                ('top',label)]
        if phase is not None:
            p+=[('include',[('phase',phase)])]

        transform_param=[('scale',scale)]
        if mean_file is not None:
            transform_param+=[('mean_file',mean_file)]
        for i in mean_value:
            transform_param+=[('mean_value',i)]
        p+=[('transform_param',transform_param)]
            
        if backend=="LMDB":
            p+=[
            ('data_param',[
               ('source',source),
               ('batch_size',batch_size),
               ('backend',backend)
           ])]
        elif backend=='image' or backend=='images':
            image_data_param=[
                    ('source',source),
                    ('batch_size',batch_size),
                    ('is_color',is_color),
                    ('shuffle',False)
                ]            
            if new_height is not None and new_width is not None:
                image_data_param+=[('new_height',new_height),('new_width',new_width)]
            p+=[('image_data_param',image_data_param)]
        else:
            raise Exception("no implementation")
        self.__end(p,name)

    def Input(self,height,width,scale=1,name="data"):
        p=[('name',name),('type','Input'),('top',name),]
        self.__end(p,name)
    #-------------------------Core----------------------------------
    def Convolution(self,
        name,
        num_output,
        bottom=None,
        kernel_size=3,
        pad=1,
        weight_filler="msra",
        dilation=None,
        stride=1):
        name=self.__start(name,'conv')

        if bottom is None:
            bottom=self.bottom
        conv_param=[
               ('num_output',num_output),
               ('pad',pad),
               ('kernel_size',kernel_size),
               ('stride',stride),
               ('weight_filler',[
                   ('type',weight_filler)
               ])
           ]
        if dilation is not None:
            conv_param+=[('dilation',dilation)]
        p=[('name',name),
           ('type',"Convolution"),
           ('bottom',bottom),
           ('top',name),
           ('param',[
               ('lr_mult',1),
               ('decay_mult',1)
           ]),
           ('param',[
               ('lr_mult',2),
               ('decay_mult',0)
           ]),
           ('convolution_param',conv_param)
           ]
        self.proto+=self.__printList(p)
        self.bottom=name
    
    def Deconvolution(self, name, num_output, bottom=None, stride=2,kernel_size=2, weight_filler="msra"):
        if bottom is None:
            bottom=self.bottom
        name=self.__start(name,'up')
        conv_param=[
               ('num_output',num_output),
               ('kernel_size',kernel_size),
               ('stride',stride),
               ('bias_filler',[
                   ('type','constant')
               ])
           ]
        weight_filler_list=[('type',weight_filler)]
        if weight_filler=='gaussian':
            weight_filler_list.append(('std',0.0001))
        conv_param.append(('weight_filler',weight_filler_list))
        
        p=[('name',name),
           ('type',"Deconvolution"),
           ('bottom',bottom),
           ('top',name),
           ('param',[
               ('lr_mult',1),
               ('decay_mult',1)
           ]),
           ('param',[
               ('lr_mult',2),
               ('decay_mult',0)
           ]),
           ('convolution_param',conv_param)
           ]
        self.__end(p,name)
    
    def InnerProduct(self,name, num_output, bottom=None, weight_filler="xavier"):
        if bottom is None:
            bottom=self.bottom
        name=self.__start(name,"dense")
        p=[('name',name),
           ('type',"InnerProduct"),
           ('bottom',bottom),
           ('top',name),
           ('param',[
               ('lr_mult',1),
               ('decay_mult',1)
           ]),
           ('param',[
               ('lr_mult',2),
               ('decay_mult',0)
           ]),
           ('inner_product_param',[
               ('num_output',num_output),
               ('weight_filler',[
                   ('type',weight_filler)
               ]),
               ('bias_filler',[
                   ('type',"constant"),
                   ('value',0)
               ])
           ])]
        self.__end(p,name)



    def ReLU(self,name):
        name=self.__start(name,'relu')
        p=[('name',name),
           ('type',"ReLU"),
           ('bottom',self.bottom),
           ('top',self.bottom)]
        self.proto+=self.__printList(p)
    
    def Pooling(self,name,pool="MAX",bottom=None, kernel_size=2,stride=2, global_pooling=False):
        """pooling and global_pooling
        :param pool: MAX,AVE,STOCHASTIC
        """
        if bottom is None:
            bottom=self.bottom
        name=self.__start(name,'pool')
        p=[('name',name),
           ('type','Pooling'),
           ('bottom',bottom),
           ('top',name)]
        
        pooling_param=[('pool',pool)]
        if global_pooling:
            pooling_param+=[('global_pooling',True)]
        else:
            pooling_param+=[
               ('kernel_size',kernel_size),
               ('stride',stride)
            ]
        p+=[('pooling_param',pooling_param)]
        self.__end(p,name)


    
    def Dropout(self,name,omit=0.5):
        name=self.__start(name,'drop')
        p=[('name',name),
           ('type',"Dropout"),
           ('bottom',self.bottom),
           ('top',self.bottom)]
        p+=[('dropout_param',[
            ('dropout_ratio',omit)
        ])]
        p+=[('include',[
            ('phase','TRAIN')
        ])]
        self.proto+=self.__printList(p)

    def Python(self,module,layer, name=None,bottom=[],top=[], other_params=[]):
        if name is None:
            name=module
        p=[('name',name),
           ('type','Python')]
        for i in bottom:
            p+=[('bottom',i)]
        for i in top:
            p+=[('top',i)]
        p+=[('python_param',[
            ('module',module),
            ('layer',layer)
        ])]
        for param in other_params:
            p+=[(param[0],param[1])]
        self.__end(p,name)
    #-----------------------------combinations---------------------------- 
    def conv_relu(self,conv,relu,
            num_output,
            bottom=None,
            kernel_size=3,
            pad=1,
            weight_filler="msra",
            dilation=None,
            stride=1):
        self.Convolution(conv,num_output,bottom,kernel_size,pad,weight_filler,dilation, stride)
        self.ReLU(relu)
    
    def fc_relu(self, name, relu, num_output, bottom=None, weight_filler="xavier"):
        self.InnerProduct(name, num_output, bottom, weight_filler)
        self.ReLU(relu)

    def fc_relu_dropout(self, name, relu, dropout, num_output, bottom=None, weight_filler="xavier", keep=.5):
        self.fc_relu(name, relu, num_output, bottom, weight_filler)
        self.Dropout(dropout,keep)

    #-------------------------Loss----------------------------------
    def Sigmoid(self,name="prob",bottom=None):
        if bottom is None:
            bottom=self.bottom
        p=[('name',name),
           ('type','Sigmoid'),
           ('bottom',bottom),
           ('top',name)]

        self.__end(p,name)

    def diceLoss(self,pred,label):
        name="loss"
        p=[('name',name),
           ('type',"Python"),
           ('top',"loss"),
           ('bottom',pred),
           ('bottom',label),
           ('python_param',[
               ('module',"perClassLoss"),
               ('layer',"perClassLossLayer")
           ]),
           ('loss_weight',1)]

        self.__end(p,name)

    def SoftmaxWithLoss(self,name="softmaxwithloss",bottom2='label',bottom1=None, loss_weight=1):
        if bottom1 is None:
            bottom=self.bottom
        p=[('name',name),
           ('type','SoftmaxWithLoss'),
           ('bottom',bottom1),
           ('bottom',bottom2),
           ('top',name),
           ('loss_weight',loss_weight)]

        self.__end(p,name)

    #-------------------------operation----------------------------------
    def Concat(self,name,bottom1,bottom2, top=None,axis=1):
        name=self.__start(name,'concat')
        if top is None:
            top=name
        p=[('name',name),
           ('type','Concat'),
           ('bottom',bottom1),
           ('bottom',bottom2),
           ('top',top),
           ('concat_param',[
               ('axis',axis)
           ])]
        self.__end(p,name)

    def silence(self,*bottom):
        p=[('name','silence'),('type','Silence')]
        for i in bottom:
            p+=[('bottom',i)]
        self.proto+=self.__printList(p)

    #-------------------------private----------------------------------


    def __start(self,name,Type):
        if isinstance(name,int):
            name=Type+str(name)
        elif isinstance(name,list):
            tmp=Type
            for i in name:
                tmp+='_'+str(i)
            name=tmp
        return name

    def __end(self,p,name):
        self.proto+=self.__printList(p)
        self.bottom=name
        
    def totxt(self, filename):
        with open(filename,"w") as f:
            f.write(self.proto)
    

    def __printList(self,List,depth=0):
        if depth==0:
            return 'layer {\n'+self.__printList(List,depth=1)+'}\n\n'
        else:
            ret=''
            for i in List:
                if isinstance(i[1], list):
                    ret+=' '*2*depth+i[0]+' {\n'+self.__printList(i[1],depth=depth+1)+' '*2*depth+'}\n'
                else:
                    field=i[1]
                    if isinstance(field,bool):
                        if field==True:
                            field='true'
                        elif field==False:
                            field='false'
                    
                    if isinstance(field, str):
                        if i[0] in self.string_field:
                            ret+=' '*2*depth+i[0]+': "'+field+'"\n'
                        else:
                            ret+=' '*2*depth+i[0]+': '+field+'\n'
                    else:
                        ret+=' '*2*depth+i[0]+': '+str(field)+'\n'
            return ret

        
 
        
        
        
        
                    

        
