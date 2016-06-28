# Yihui He, https://yihui-he.github.io
from __future__ import print_function
import numpy as np
import cv2
import caffe
from utils import Data, NetHelper
import cfgs
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv)==1:
    debug=False
else:
    debug=int(sys.argv[1])


def classifier(c_img, nh,thresh=0.999,showIm=True):
    pred=nh.bin_pred_map(c_img)
    pred_bin=pred.copy()
    pred_bin[pred>thresh]=1
    pred_bin[pred<=thresh]=0
    return pred_bin,pred

def prep(img):
    img = img.astype('float32')
    img = cv2.resize(img, (cfgs.inShape[1], cfgs.inShape[0]))
    return img


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    ret=' '.join([str(r) for r in res])
    return ret

def func(filename, nh):
    _,idx,ext=Data.splitPath(filename)
    if ext!=".tif": 
        return None
    cfgs.cnt+=1
    print(cfgs.cnt)
    #idx=int(idx)
    img=Data.imFromFile(filename)
    pred_bin,pred=classifier(img,nh)
    result=run_length_enc(prep(pred_bin))
    if debug:
        hist=np.histogram(pred)
        print(pd.DataFrame(hist[0],index=hist[1][1:]).T)
        mask=plt.imread(os.path.join(cfgs.train_mask_path,idx+"_mask.tif"))
        plt.figure(1)
        plt.subplot(221)
        plt.imshow(mask)
        plt.subplot(222)
        plt.imshow(pred_bin)
        plt.subplot(223)
        plt.imshow(img)
        plt.subplot(224)
        plt.imshow(pred)
        plt.show()
        print(idx,result)

    return (idx,result)

def submission():

    NetHelper.gpu(1)
    #submission()
    nh=NetHelper(deploy=cfgs.deploy_pt,model=cfgs.best_model_dir)
    if debug:
        l=Data.folder_opt(cfgs.train_data_path,func,nh)
    else:
        l=Data.folder_opt(cfgs.test_data_path,func,nh)
    l=np.array(l,dtype=[('x',int),('y',object)])
    l.sort(order='x')

    first_row = 'img,pixels'
    file_name = 'submission.csv'

    with open(file_name, 'w+') as f:
        f.write(first_row)
        for i in l:
            s = str(i[0]) + ',' + i[1]
            f.write(('\n'+s))

def testSingleImg():
    NetHelper.gpu()
    #submission()
    nh=NetHelper(deploy=cfgs.deploy_pt,model=cfgs.best_model_dir)
    img=Data.imFromFile(os.path.join(cfgs.train_mask_path,"1_1_mask.tif"))
    res=nh.bin_pred_map(img)
    print(np.histogram(res))
    

if __name__ == '__main__':
    submission()
    
    