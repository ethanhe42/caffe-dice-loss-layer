#-.-encoding=utf-8-.-``
# Yihui He, https://yihui-he.github.io
''' My caffe helper'''
import caffe
import os
import warnings
import cv2
import numpy as np
import PIL.Image as Image

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
    def folder_opt(folder, func):
        """A func operate on each image then return a list"""
        all_list=[]
        for img in os.listdir(folder):
            if len(img) == 0:
                raise ValueError("invalid name")
            if len(img.replace(' ','')) != len(img):
                warnings.warn("whitespace in name")
            filename = os.path.join(folder, img)
            all_list.append(func(filename))
        return all_list

class NetHelper:
    """Helper for dealing with net"""
    def __init__(self, net):
        self.net=net

    @staticmethod
    def netFromFile(deploy_file,model_file,mode=caffe.TEST):
        return caffe.Net(deploy_file,model_file,mode)

    def prediction(self,c_img):
        """make prediction on single img"""
        if len(c_img.shape)==3:
            # color img
            pass
        else:
            # grey img
            tmp = np.uint8(np.zeros(c_img[:,:,np.newaxis].shape))
            tmp[:,:,0]=c_img
        c_img  = c_img.swapaxes(1,2).swapaxes(0,1) 
        # in Channel x Height x Width order (switch from H x W x C)
        c_img  = c_img.reshape((1,3,c_img.shape[1],c_img.shape[2]))
        # 1 means batch size one
        prediction = self.net.forward_all(**{net.inputs[0]: c_img})
        return prediction

    def bin_pred_map(self,c_img, last_layer='prob',prediction_map=1):
        """get binary probability map prediction"""
        pred=self.prediction(c_img)
        prob_map=np.single(pred[last_layer][0,prediction_map,:,:])
        return prob_map
    
    def inspect(self):
        """inspect network params"""
        pass