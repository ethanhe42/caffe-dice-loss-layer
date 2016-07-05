#-.-encoding=utf-8-.-``
# Yihui He, https://yihui-he.github.io
import sys
import pandas as pd
sys.path.append("/home/yihuihe/Ultrasound-Nerve-Segmentation")
# sys.path.insert(0, "/home/yihuihe/deeplab-public-ver2/python")
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

# gen solver prototxt
solver=CaffeSolver(debug=cfgs.debug)
solver.sp=cfgs.sp.copy()
solver.write(cfgs.solver_pt)

debug=True

weights = cfgs.init

# init
caffe.set_device(2)
caffe.set_mode_gpu()
# caffe.set_mode_cpu()

solver = caffe.SGDSolver(cfgs.solver_pt)
# solver.net.copy_from(weights)

for iter in range(500*2000):
    if debug:
        if iter % 100 == 0:
            nethelper=NetHelper(solver.net)
            nethelper.hist('prob', filters=2,attr="blob")
            # TODO: label has float
            nethelper.hist('label', filters=2,attr="blob")
            
    solver.step(1)
