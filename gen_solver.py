''' genrate my solver'''
from utils import CaffeSolver
import cfgs
import os

model="deeplab"
tr_pt="trainval.prototxt"
te_pt="test.prototxt"

solver=CaffeSolver(debug=True)

solver.testnet_prototxt_path=os.path.join(model,tr_pt)
solver.trainnet_prototxt_path=os.path.join(model,te_pt)
solver.sp['average_loss']=20
solver.sp['lr_policy']="poly"
solver.sp['power']=.9
solver.sp['base_lr']=1e-6
solver.sp['max_iter']=200000
solver.sp['momentum']=.9
solver.sp['weight_decay']=0.0005
solver.sp['test_initialization']= "false"
solver.sp['snapshot']=5000
solver.sp['snapshot_prefix']=os.path.join(cfgs.usr_dir,model)

#test
solver.sp['test_iter']=20
solver.sp['test_interval']=40
solver.sp['display']=20

