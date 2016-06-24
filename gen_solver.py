''' genrate my solver'''
from utils import CaffeSolver
import cfgs
import os

model="deeplab"
tr_pt="trainval.prototxt"
te_pt="test.prototxt"
solver_name="solver.prototxt"

solver=CaffeSolver(debug=True)
solver.sp['average_loss']=20
solver.sp['lr_policy']="poly"
solver.sp['power']=.9
solver.sp['base_lr']=1e-6
solver.sp['max_iter']=200000
solver.sp['momentum']=.9
solver.sp['weight_decay']=0.0005
solver.sp['test_initialization']= True
solver.sp['snapshot']=5000
#test
solver.sp['test_iter']=20
solver.sp['test_interval']=40
solver.sp['display']=20

solver.sp['snapshot_prefix']=os.path.join('/mnt/data1/yihuihe/ultrasound-nerve')
solver.sp['train_net']=os.path.join(model,tr_pt)
solver.sp['test_net']=os.path.join(model,te_pt)
solver.write(os.path.join(model,solver_name))


