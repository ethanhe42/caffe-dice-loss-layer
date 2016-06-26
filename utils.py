#-.-encoding=utf-8-.-``
# Yihui He, https://yihui-he.github.io
''' My caffe helper'''
import caffe
import os
import warnings
import cv2
import numpy as np
import PIL.Image as Image
import sys
import lmdb
import random

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
        self.sp['lr_policy'] = 'fixed' # poly steps, see caffe proto

        # important, but rare:
        self.sp['gamma'] = 1 # If learning rate policy: drop the learning rate in "steps" by a factor of gamma every stepsize iterations drop the learning rate by a factor of gamma
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
            all_list.append(func(filename, *args))
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
            tmp = np.uint8(np.zeros(c_img[:,:,np.newaxis].shape))
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
    
    def hist(self,layer,bins=10):
        """inspect network params via histogram"""
        response=self.net.blobs[layer].data
        cnts,boundary = np.histogram(response.flatten(),bins=bins)
        print layer,cnts
        print boundary