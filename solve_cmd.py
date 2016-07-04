import subprocess
import sys
import os
subprocess.check_call(["export","PYTHONPATH=$PYTHONPATH:"+os.getcwd()])
subprocess.check_call(["/home/yihuihe/deeplab-public-ver2/build/tools/caffe","train","-solver","/home/yihuihe/Ultrasound-Nerve-Segmentation/unet/solver.prototxt","-gpu","all"])