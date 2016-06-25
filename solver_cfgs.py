import os
model="deeplab"
tr_pt="trainval.prototxt"
te_pt="test.prototxt"
solver_name="solver.prototxt"

debug=True
sp=dict()
sp['average_loss']=20
sp['lr_policy']="poly"
sp['power']=.9
sp['base_lr']=1e-6
sp['max_iter']=200000
sp['momentum']=.9
sp['weight_decay']=0.0005
sp['test_initialization']= True
sp['snapshot']=5000
#test
if debug:
    sp['test_iter']=20
    sp['test_interval']=40
    sp['display']=20

sp['snapshot_prefix']=os.path.join('/mnt/data1/yihuihe/ultrasound-nerve')
sp['train_net']=os.path.join(model,tr_pt)
sp['test_net']=os.path.join(model,te_pt)

