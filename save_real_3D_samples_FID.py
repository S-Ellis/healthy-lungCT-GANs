#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Sam Ellis
Copyright 2022 Sam Ellis, King's College London
"""
import argparse
import os
import torch
import torch.utils.data
import numpy as np
import patch_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--patches_per_CT', type=int, default=20,help='Number of patches to load per CT image')
parser.add_argument('--manualSeed', type=int, help='Manual seed')
parser.add_argument('--outf', default='real_samples_3D_FID/', help='Folder to output real images for fid calculation')

opt = parser.parse_args()
print(opt)
 
#%%
lungSegListCT = np.atleast_1d(np.loadtxt('lungSegList_train.txt','str'))
        
imageListCT = np.atleast_1d(np.loadtxt('imageList_train.txt','str'))                
                
# sanity check, the two vectors should be the same filenames
if not [os.path.basename(lungSegListCT[ii]) for ii in range(len(lungSegListCT))] == [os.path.basename(imageListCT[ii]) for ii in range(len(imageListCT))]:
    raise ValueError('The file lists do not match')
        
dataset = patch_dataset.patchLoader(imageListCT, lungSegListCT, num_patch_per_CT = opt.patches_per_CT)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True, num_workers=0) # load in patches from a single image at a time
    
#%%
try:
    outf = opt.outf
    os.makedirs(outf)
except OSError:
    pass

#%% perform the sampling
with torch.no_grad():
    for i, data_full in enumerate(dataloader, 0):
        print(i)
        data_full = data_full[0]
        
        data_full = (data_full + 1.) /2.
                
        torch.save(data_full,f'{outf}real_batch_{i:05d}.pth')



    
    
    