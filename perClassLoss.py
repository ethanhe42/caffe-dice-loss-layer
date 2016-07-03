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
        self.diff=np.zeros_like(bottom[0].data,dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...]=bottom[1].data
        self.sum=bottom[0].data.sum()+bottom[1].data.sum()+1.
        self.dice=(2.* (bottom[0].data * bottom [1].data).sum()+1.)/self.sum
        top[0].data[...] = 1.- self.dice

    def backward(self, top, propagate_down, bottom):
        if propagate_down[1]:
            raise Exception("label not diff")
        elif propagate_down[0]:
            bottom[0].diff[...] = (-2*self.diff+self.dice)/self.sum
        else:
            raise Exception("no diff")


