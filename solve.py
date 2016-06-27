#-.-encoding=utf-8-.-``
# Yihui He, https://yihui-he.github.io
import sys
import pandas as pd
sys.path.append("/home/yihuihe/Ultrasound-Nerve-Segmentation")
sys.path.insert(0, "/home/yihuihe/deeplab-public-ver2/python")
print sys.path
import caffe
print caffe.__file__
import numpy as np
import cv2
from utils import NetHelper, CaffeSolver
import cfgs
import score
import surgery
import os

# gen solver prototxt
solver=CaffeSolver(debug=cfgs.debug)
solver.sp=cfgs.sp.copy()
solver.write(cfgs.solver_pt)

debug=True
inspect_layers=['geo_shrink','loss_geo','convf']
# import setproctitle
# setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = cfgs.init

# init
caffe.set_device(int(1))
caffe.set_mode_gpu()
# caffe.set_mode_cpu()

solver = caffe.SGDSolver(cfgs.solver_pt)
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
#test = np.loadtxt('../data/sift-flow/test.txt', dtype=str)

for iter in range(50*2000):
    if debug:
        if iter % 400 == 0:
            nethelper=NetHelper(solver.net)
            nethelper.hist('convf', filters=True)
            
    solver.step(1)
    # N.B. metrics on the semantic labels are off b.c. of missing classes;
    # score manually from the histogram instead for proper evaluation
    # score.seg_tests(solver, False, test, layer='score_geo', gt='geo')
