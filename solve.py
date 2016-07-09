#-.-encoding=utf-8-.-``
# Yihui He, https://yihui-he.github.io
import sys
import pandas as pd
sys.path.append("/home/yihuihe/Ultrasound-Nerve-Segmentation")
sys.path.insert(0, "/home/yihuihe/miscellaneous/caffe/python")
print sys.path
import caffe
print caffe.__file__
import numpy as np
import cv2
from utils import NetHelper, CaffeSolver
# import cfgs
import unet_cfgs as cfgs
#import score
#import surgery
import os
import matplotlib.pyplot as plt

# gen solver prototxt
solver=CaffeSolver(debug=cfgs.debug)
solver.sp=cfgs.sp.copy()
solver.write(cfgs.solver_pt)

debug=True

weights = cfgs.init

# init
caffe.set_device(3)
caffe.set_mode_gpu()
# caffe.set_mode_cpu()

solver = caffe.SGDSolver(cfgs.solver_pt)
if weights is not None:
    solver.net.copy_from(weights)

for iter in range(500*2000):
    if debug:
        if iter % 100 == 0 and iter !=0:
            nethelper=NetHelper(solver.net)
            # nethelper.hist('data')
            nethelper.hist('label')
            nethelper.hist('prob', filters=2,attr="blob")
            nethelper.hist('data', filters=2,attr="blob")

            if False:
                for i in range(nethelper.net.blobs['data'].data.shape[0]):
                    plt.subplot(221)
                    plt.imshow(nethelper.net.blobs['data'].data[i,0])
                    plt.subplot(222)
                    plt.imshow(nethelper.net.blobs['prob'].data[i,0])
                    plt.subplot(223)
                    plt.imshow(nethelper.net.blobs['label'].data[i,0])
                    plt.show()
                
            

            # TODO: label has float
            # nethelper.value_counts('label')
            
    solver.step(1)
