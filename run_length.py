#-.-encoding=utf-8-.-``
# Yihui He, https://yihui-he.github.io

import numpy as np
import cv2
import caffe
from utils import Data, NetHelper
import cfgs
import os
from PIL import Image
import pandas as pd

debug=False
def classifier(c_img, net,thresh=0.1):
    pd=NetHelper(net).bin_pred_map(c_img)
    print sum(sum(pd))
    pd[pd>thresh]=1
    pd[pd<=thresh]=0
    return pd

def prep(img):
    img = img.astype('float32')
    img = cv2.resize(img, (cfgs.inShape[0], cfgs.inShape[0]))
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
    return ' '.join([str(r) for r in res])

def func(filename, net):
    _,idx,_=Data.splitPath(filename)
    idx=int(idx)
    img=Data.imFromFile(filename)
    predi=classifier(img,net)
    result=run_length_enc(prep(predi))
    print idx,sum(sum(predi)),result

    return (idx,result)

def submission():

    for i in range(total):
        img = imgs_test[i, 0]
        img = prep(img)
        rle = run_length_enc(img)

        rles.append(rle)
        ids.append(imgs_id_test[i])

        if i % 100 == 0:
            print i

    first_row = 'img,pixels'
    file_name = 'submission.csv'

    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')


    

if __name__ == '__main__':
    #submission()
    img=Data.imFromFile(os.path.join(cfgs.train_data_path,"1_1.tif"))
    Data.showIm(img)
    #net=NetHelper.netFromFile(cfgs.deploy_pt,cfgs.best_model_dir)
    #img=classifier(img,net,0.05)

    cv2.imwrite(os.path.join(cfgs.test_pred_path,"40.png"),img*255)
    # Data.folder_opt(cfgs.test_data_path,func,net)

    
    