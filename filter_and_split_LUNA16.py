#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Sam Ellis
Copyright 2022 Sam Ellis, King's College London
"""
import argparse
import os
import numpy as np
import pylidc as pl # module for handling the LIDC/LUNA16 dataset

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str,default='', help="Path to LUNA16 folder")
parser.add_argument('--lung_seg_path', type=str,default='', help="Path to LUNA16 lung segmentations folder")
parser.add_argument('--malignantThresh', type=int, default=4, help='For the purpose of training a healthy GAN, we disregard CTs if the annotated malignancy is >= malignantThresh')
[opt,_] = parser.parse_known_args()

# first, get the list of all series UIDs for the segmentations
seg_list = os.listdir(opt.lung_seg_path)
lungSegListCT = []
series_uids = []
for file in seg_list:
    if file[-4::]=='.mhd':
        series_uids.append(file[0:-4:])
        lungSegListCT.append(opt.lung_seg_path + file)
series_uids = np.sort(series_uids)
lungSegListCT = np.sort(lungSegListCT)
        
# now, for eash series UID, find it in the data path
imageListCT = []
for series_uid in series_uids:
    for root, dirnames, filenames in os.walk(opt.image_path):
        for filename in filenames:
            if filename.endswith(f'{series_uid}.mhd'):
                imageListCT.append(os.path.join(root, filename))
imageListCT = np.array(imageListCT)


#%%
malignantFlagsMedianScore = np.zeros((len(imageListCT)))
for idx in range(len(imageListCT)):
    print(f'{idx} of {len(imageListCT)}')

    image_file = imageListCT[idx]
    series_uid_im = os.path.basename(image_file)[0:-4]
    
    seg_file = lungSegListCT[idx]
    series_uid_seg = os.path.basename(image_file)[0:-4]        
    
    # check that the series_uids match!
    assert series_uid_im==series_uid_seg
    
    series_uid = series_uid_im
    
    # get the nodule information from pylidc and generate a nodule mask image which is the same size as the original image
    scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == series_uid)[0]
    
    # loop over each nodule
    nod_pls = scan.cluster_annotations()
    for anns in nod_pls:
        allNodMalignancies = []
        for ann in anns:
            allNodMalignancies.append(ann.malignancy)
        
        allNodMalignancies = np.array(allNodMalignancies)
        if np.median(allNodMalignancies) >= opt.malignantThresh:
            malignantFlagsMedianScore[idx] = 1.0
        
print(f'{np.count_nonzero(malignantFlagsMedianScore)} patients rejected due to have at least one nodule with median malignancy >= {opt.malignantThresh} (out of {len(imageListCT)} patients)')

#%% get list of files, and split in train test
lungSegListCT = lungSegListCT[np.where(malignantFlagsMedianScore==0)]
imageListCT = imageListCT[np.where(malignantFlagsMedianScore==0)]

assert len(imageListCT) == len(lungSegListCT)

# shuffle in unison
p = np.random.permutation(len(imageListCT))
imageListCT = imageListCT[p]
lungSegListCT = lungSegListCT[p]

num_scans_accepted = len(imageListCT)

# split into train and val/test and write to files
num_train = np.ceil((4 * num_scans_accepted)/5).astype('int')
num_test = num_scans_accepted - num_train

trainLungSegListCT = lungSegListCT[0:num_train]
testLungSegListCT =  lungSegListCT[num_train::]

trainImageListCT = imageListCT[0:num_train]
testImageListCT =  imageListCT[num_train::]

np.savetxt(f'lungSegList_train.txt',trainLungSegListCT,fmt="%s")
np.savetxt(f'imageList_train.txt',trainImageListCT,fmt="%s")

np.savetxt(f'lungSegList_test.txt',testLungSegListCT,fmt="%s")
np.savetxt(f'imageList_test.txt',testImageListCT,fmt="%s")