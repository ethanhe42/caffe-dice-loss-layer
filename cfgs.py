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

# prototxt
pt_folder="deeplab"
tr_pt=os.path.join(pt_folder,"trainval.prototxt")
te_pt=os.path.join(pt_folder,"test.prototxt")
solver_pt=os.path.join(pt_folder,"solver.prototxt")
deploy_pt=os.path.join(pt_folder,"deploy.prototxt")

# saved model
init='/home/yihuihe/medical-image-segmentation/deeplab/init.caffemodel'
model_name="ultrasound-nerve"
model_save_path="/mnt/data1/yihuihe"
best_model=5
best_model_dir=os.path.join(model_save_path,
    model_name+'_iter_'+str(best_model)+'000.caffemodel')

# solver
model="deeplab"

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
if debug:
    sp['test_iter']=20
    sp['test_interval']=1000
    sp['display']=20
sp['snapshot_prefix']=os.path.join('/mnt/data1/yihuihe',model_name)
sp['train_net']=tr_pt
sp['test_net']=te_pt

# net
inShape=(420,580,1)
outShape=(420,580,1)



#-----------------------------------Global vars
cnt=0

