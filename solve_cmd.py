import subprocess
import sys
import os
# subprocess.check_call(["export","PYTHONPATH=$PYTHONPATH:"+os.getcwd()])
subprocess.check_call(["/home/yihuihe/deeplab-public-ver2/build/tools/caffe","train","-solver","/home/yihuihe/Ultrasound-Nerve-Segmentation/deeplab_res/solver.prototxt","-gpu","0","-weights","/mnt/data1/yihuihe/deeplab/train2_iter_20000.caffemodel"])