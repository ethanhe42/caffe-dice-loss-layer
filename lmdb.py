# create lmdb files 
#-.-encoding=utf-8-.-``
# whenever use Image.open, use 'try'
# convert both raw and mask images into lmdb files
# TODO: Selectively build lmdb with focus

''' 
input: output_train.txt or output_val.txt
output: lmdb dataset files of train or val
NOTE: label has already been resized to their targetedSize, like say 256x256 in this stage 
'''

import caffe
# import lmdb
from PIL import Image
import numpy as np
import os
import cv2
import sys


debug=False # for overfitting of 4 images
''' image resizing using PIL'''
def imresize(im,sz):
    pil_im = Image.fromarray(np.uint8(im))
    return np.array(pil_im.resize(sz))


''' ------------------------------------------------- '''  
''' input param '''
input_ = sys.argv[1] #'train'
#input_ = 'val'
data_root = './'
lmdb = 'lmdb_train_val'
''' ------------------------------------------------- '''



''' resize dim '''
sz_data = (256,256) # image data
sz_mask = (128,128)


''' path '''
save_data_root = os.path.join(data_root, lmdb)
input_lst = input_ + '.txt' # this is a txt file
#read_data_root = os.path.join(data_root, 'image_crop')
#read_mask_root = os.path.join(data_root, 'mask_crop_denoise_128')
save_im_lmdb_root = os.path.join(save_data_root, input_ + '_data')
save_mask_lmdb_root = os.path.join(save_data_root, input_ + '_mask')
if not os.path.exists(save_data_root): os.mkdir(save_data_root)
if not os.path.exists(save_im_lmdb_root): os.mkdir(save_im_lmdb_root)
if not os.path.exists(save_mask_lmdb_root): os.mkdir(save_mask_lmdb_root)


''' read train or val txt files '''
train_val_list = os.path.join(data_root, input_lst)
f = open(train_val_list, 'r')
inputs = f.read().splitlines() # read lines in file
f.close()



''' process data lmdb '''
''' ------------------------------------------------- '''
import lmdb
in_db = lmdb.open(save_im_lmdb_root, map_size=int(1e12))

with in_db.begin(write=True) as in_txn:
    # in_idx is indexing number starting from 0; in_ is actually the information
    for in_idx, in_ in enumerate(inputs):              
        if in_idx>3 and debug==True:
            break

        if in_idx%100 == 0: print "%d images have been processed" %in_idx
        path = in_
        try: 
            img = np.array(Image.open(path), dtype=np.float32) # check if opened successfully
        except:
            # print in_
            print 'image is damaged'
            continue
        # print img.shape
        # from psd we get 4-channel image, RGBA
        im = img[:,:,0:3] # ignore A channel
        im = imresize(im,sz_data)
        im = im[:,:,::-1] # in BGR (switch from RGB), opencv in the form of BGR
        im = im.swapaxes(1,2).swapaxes(0,1) # in Channel x Height x Width order (switch from H x W x C)
        im_dat = caffe.io.array_to_datum(im) # convert to caffe friendly data format
        # in_idx is the key; im_dat is the value; in_idx keeps track of the lmdb data index
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
    
    print '%d images have been processed' %(in_idx+1)

in_db.close()
print 'image lmdb dataset has been created'
''' ------------------------------------------------- '''




''' process mask lmdb'''
''' ------------------------------------------------- '''
in_db = lmdb.open(save_mask_lmdb_root, map_size=int(1e12))

with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        if in_idx>3 and debug==True:
            break

        if in_idx%100 == 0: print "%d masks have been processed" %in_idx
        path = os.path.join(os.path.dirname(in_).replace('png', 'mask'), os.path.basename(in_)) #TODO: replace is a potential bug
        print path
        im = np.array(Image.open(path), dtype=np.float32)

        ''' -------------------------------------------------- '''
        # im = imresize(img,sz_mask)
        # label resize function SHOULD NOT be achieved by 'resize' function
        # instead, it must be done by label_down_sample.py 
        ''' -------------------------------------------------- '''

        all_pos = np.where(im!=0)


        tgt_im = np.zeros((sz_mask[0], sz_mask[1]), dtype=np.uint8)

        if len(all_pos):
            tgt_im[all_pos[0] * sz_mask[0] / im.shape[0], all_pos[1] * sz_mask[1] / im.shape[1]] = 1

        #cv2.imshow('debug.png', tgt_im*255)
        #cv2.waitKey(-1)
        
        im = tgt_im
        ''' mask channel is 1 '''
        # since im has only two axis, we need to have 3 axis
        tmp = np.uint8(np.zeros(im[:,:,np.newaxis].shape)) # new a tmp image
        # print tmp.shape
        tmp[:,:,0] = im  
        # print tmp.shape
        # in Channel x Height x Width order (switch from H x W x C)
        tmp = tmp.swapaxes(1,2).swapaxes(0,1) 

        im_dat = caffe.proto.caffe_pb2.Datum()
        im_dat.channels = tmp.shape[0]
        im_dat.height = tmp.shape[1]
        im_dat.width = tmp.shape[2]
        im_dat.label = int(in_idx)
        im_dat.data = tmp.tostring()
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())

    print '%d masks have been processed' %(in_idx+1)

in_db.close()

print 'mask lmdb dataset has been created'