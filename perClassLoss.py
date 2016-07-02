#-.-encoding=utf-8-.-``
# Yihui He, https://yihui-he.github.io

""" pt 
layer {
  type: 'Python'
  name: 'loss'
  top: 'loss'
  bottom: 'ipx'
  bottom: 'ipy'
  python_param {
    module: 'perClassLoss' # the module name -- usually the filename -- that needs to be in $PYTHONPATH
    layer: 'perClassLossLayer' # the layer name -- the class name in the module
  }
  loss_weight: 1 # set loss weight so Caffe knows this is a loss layer
}
"""

import caffe
import numpy as np
import warnings

class perClassLossLayer(caffe.Layer):
    """
    self designed loss layer for segmentation. Class weighted, per pixel loss
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count!=bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.sum=bottom[0].data.sum()+bottom[1].data.sum()+1.
        self.dice=(2.* (bottom[0].data * bottom [1].data).sum()+1.)/self.sum
        top[0].data[...] = 1.- self.dice

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * (-2*bottom[1].data+self.dice)/self.sum
            print np.histogram(bottom[i].diff[...])


