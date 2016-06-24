import sys
import pandas as pd
sys.path.append("/home/yihuihe/Ultrasound-Nerve-Segmentation")
sys.path.insert(0, "/home/yihuihe/deeplab-public-ver2/python")
print sys.path
import caffe
print caffe.__file__
import numpy as np

import score
import surgery

debug=False
inspect_layers=['geo_shrink','loss_geo','convf']
# import setproctitle
# setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = '/home/yihuihe/medical-image-segmentation/deeplab/init.caffemodel'

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()
# caffe.set_mode_cpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
#test = np.loadtxt('../data/sift-flow/test.txt', dtype=str)

for _ in range(50*2000):
    if debug:
        print pd.value_counts(solver.net.blobs[inspect_layers[0]].data.flatten())
        for i in inspect_layers:
            print i, solver.net.blobs[i].data.shape
    solver.step(1)
    # N.B. metrics on the semantic labels are off b.c. of missing classes;
    # score manually from the histogram instead for proper evaluation
    # score.seg_tests(solver, False, test, layer='score_geo', gt='geo')
