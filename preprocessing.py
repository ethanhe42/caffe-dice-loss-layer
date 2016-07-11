
# imports
import caffe

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
class transformLayer(caffe.Layer):

    """
    This is a simple syncronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):
        self.nb_top=3
        self.nb_bottom=4
        self.batch_size=bottom[0].data.shape[0]
        
        for i in range(self.nb_top):
            top[i].reshape(*bottom[i].data.shape)

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
        # skip transformLayer in test phase
        if self.phase==caffe.TEST:
            for i in range(self.nb_top):
                top[i].data[...]=bottom[i].data
            return
        
        debug=False
        grid=230

        self.newdata=np.ndarray(bottom[0].data.shape)
        self.newlabel=np.ndarray(bottom[1].data.shape)
        self.hasObj=np.ndarray(bottom[2].data.shape)

        for i in range(self.batch_size):
            img=bottom[0].data[i,0]
            label=bottom[1].data[i,0]
            hasObj=bottom[2].data[i]

            if debug:
                
                plt.subplot(grid+1)
                plt.title('org')
                plt.imshow(img)
                plt.subplot(grid+2)
                plt.title('org')
                plt.imshow(label)
            
            # random flip
            if np.random.randint(2):
                img=cv2.flip(img,0)
                label=cv2.flip(label,0)
            
            if np.random.randint(2):
                img=cv2.flip(img,1)
                label=cv2.flip(label,1)
            
            if debug:
                plt.subplot(grid+3)
                plt.title('flip')
                plt.imshow(img)
                plt.subplot(grid+4)
                plt.title('flip')
                plt.imshow(label)

            # if np.random.randint(10)!=0:
            #     im_merge = np.concatenate((img[...,None], label[...,None]), axis=2)
            #     im_merge_t = elastic_transform(im_merge, im_merge.shape[0] * 2, im_merge.shape[0] * 0.08, im_merge.shape[0] * 0.08)
            #     img= im_merge_t[...,0]
            #     label= im_merge_t[...,1]

            if debug:
                plt.subplot(grid+5)
                plt.title('elastic')
                plt.imshow(img)
                plt.subplot(grid+6)
                plt.title('elastic')
                plt.imshow(label)
                plt.show()

            self.newdata[i,0]=img
            self.newlabel[i,0]=label
            self.hasObj[i]=hasObj
        
        top[0].data[...]=self.newdata
        top[1].data[...]=self.newlabel
        top[2].data[...]=self.hasObj
            
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


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
#     for i in range(2):
#         plt.imshow(image[:,:,i])
#         plt.show()    
    randoms=random_state.rand(*shape) * 2 - 1 
    dx = gaussian_filter(randoms, sigma) * alpha
    randoms=random_state.rand(*shape) * 2 - 1

    dy = gaussian_filter(randoms, sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


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
