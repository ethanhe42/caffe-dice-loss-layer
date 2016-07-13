''' settings '''
import os
# data
data_path = "/mnt/data1/yihuihe/ultrasound-nerve"
train_data_path=os.path.join(data_path,"train_data")
train_mask_path=os.path.join(data_path,"train_mask")
test_data_path=os.path.join(data_path,"Ultrasound-Nerve-Segmentation-test")
test_pred_path=os.path.join(data_path,"test_mask")
data_list_path="data"
lmdb_path=data_path
usr_dir="home/yihuihe"
proj=os.path.join(usr_dir,"Ultrasound-Nerve-Segmentation")

# prototxt
pt_folder="deeplab_res"
tr_pt=os.path.join(pt_folder,"train.prototxt")
te_pt=os.path.join(pt_folder,"train.prototxt")
solver_pt=os.path.join(pt_folder,"solver.prototxt")
deploy_pt=os.path.join(pt_folder,"deploy.prototxt")

# saved model
model_name="deep_lab"
model_save_path="/mnt/data1/yihuihe"
best_model=5000
best_model_dir=os.path.join(model_save_path,
    model_name+'_iter_'+str(best_model)+'.caffemodel')
init='/mnt/data1/yihuihe/deeplab/train2_iter_20000.caffemodel'
# init=best_model_dir

# solver
model="deeplab"

debug=True
sp=dict()
sp['iter_size']=16
sp['average_loss']=20
sp['lr_policy']="poly"
sp['power']=.9
if sp['lr_policy']=="step":
    sp['gamma']=.1
    sp['stepsize']=300
    sp['power']=1.0
sp['base_lr']=2.5e-4
sp['max_iter']=21000
sp['momentum']=.9
sp['weight_decay']=0.0005
sp['test_initialization']= True
sp['snapshot']=10000
if debug:
    sp['test_iter']=20
    sp['test_interval']=1000
    sp['display']=20
sp['snapshot_prefix']=os.path.join('/mnt/data1/yihuihe',model_name)
sp['train_net']=tr_pt
sp['test_net']=te_pt

# net
inShape=(384,512,1)
outShape=(420,580,1)



#-----------------------------------Global vars
cnt=0

