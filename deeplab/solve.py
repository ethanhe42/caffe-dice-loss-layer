import sys
sys.path.append("/home/yihuihe/medical-image-segmentation")
sys.path.insert(0, "/home/yihuihe/deeplab-public-ver2/python")
print sys.path
import caffe
print caffe.__file__
import numpy as np

import score
import surgery

debug=False
# import setproctitle
# setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = 'init.caffemodel'

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
test = np.loadtxt('../data/sift-flow/test.txt', dtype=str)

for _ in range(50*2000):
    solver.step(1)
    if debug:
        for i in solver.net.blobs:
            print i,solver.net.blobs[i].data
    # N.B. metrics on the semantic labels are off b.c. of missing classes;
    # score manually from the histogram instead for proper evaluation
    # score.seg_tests(solver, False, test, layer='score_geo', gt='geo')
