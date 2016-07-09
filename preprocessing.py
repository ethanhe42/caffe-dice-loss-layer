
# imports
import caffe

import numpy as np
import cv2

class transformLayer(caffe.Layer):

    """
    This is a simple syncronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):
        self.nb_top=3
        self.nb_bottom=4
        

        assert len(top)==self.nb_top
        assert len(bottom) == self.nb_bottom

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        # params = eval(self.param_str)

        # Check the paramameters for validity.
        # check_params(params)

        # store input as class variables
        # self.batch_size = params['batch_size']

        # print_info("PascalMultilabelDataLayerSync", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        
        for i in range(self.nb_top):
            self.top[i]=np.ndarray(bottom[i].data.shape)
        
        for i in range(self.top[0].shape[0]):
            img=self.bottom[0].data[i,0]
            label=self.bottom[1].data[i,0]
            hasObj=self.bottom[2].data[i,0]

            if np.random.randint(2):
                img=cv2.flip(img,0)
                label=cv2.flip(label,0)
            
            if np.random.randint(2):
                img=cv2.flip(img,1)
                label=cv2.flip(label,1)
            
            self.top[0][i,0]=img
            self.top[1][i,0]=label
            self.top[2][i]=hasObj
        
        for i in range(self.nb_top):
            top[i].data[...]=self.top[i]
            
    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass

def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'pascal_root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Ouput some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['split'],
        params['batch_size'],
        params['im_shape'])
