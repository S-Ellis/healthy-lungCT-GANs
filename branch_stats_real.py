#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Sam Ellis
Copyright 2022 Sam Ellis, King's College London
"""
import argparse
import os
import numpy as np
import torch
import patch_dataset
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from skimage.transform import rescale
from scipy.ndimage import convolve
from skimage import measure
import random

parser = argparse.ArgumentParser()
parser.add_argument('--patches_per_CT', type=int, default=20,help='Number of patches to load per CT image')
parser.add_argument('--number_CTs', type=int, default=-1,help='Number of CT images to use (default is all)')
parser.add_argument('--manualSeed', type=int, help='Manual seed')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
    
#%%
lungSegListCT = np.atleast_1d(np.loadtxt('lungSegList_train.txt','str'))
        
imageListCT = np.atleast_1d(np.loadtxt('imageList_train.txt','str'))                
                
# sanity check, the two vectors should be the same filenames
if not [os.path.basename(lungSegListCT[ii]) for ii in range(len(lungSegListCT))] == [os.path.basename(imageListCT[ii]) for ii in range(len(imageListCT))]:
    raise ValueError('The file lists do not match')
    
# truncate list to get just some CTs
if opt.number_CTs < 1:
    # use all if not specified
    opt.number_CTs = len(imageListCT)
    
imageListCT = imageListCT[0:opt.number_CTs]
lungSegListCT = lungSegListCT[0:opt.number_CTs]
        
dataset = patch_dataset.patchLoader(imageListCT, lungSegListCT, num_patch_per_CT = opt.patches_per_CT)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True, num_workers=0) # load in patches from a single image at a time
    
#%% perform the sampling
all_num_branches = []
with torch.no_grad():
    for i, data_full in enumerate(dataloader, 0):
        print(i)
        data_full = data_full[0]
        
        data_full = (data_full + 1.) /2.
        
        # loop over each image
        for jj in range(data_full.size(0)):
            curr_im = np.squeeze(data_full[jj, :, :, :, :].numpy())
               
            # upsample to isotropic resolution
            curr_im = rescale(curr_im, (2,1,1), anti_aliasing=False,order=1)

            # binarise            
            thresh = threshold_otsu(curr_im)
            binary = (curr_im >= thresh).astype('uint8')
            
            # skeletonise
            skeleton = skeletonize(binary)
 
            # detect branch points
            kernel = np.ones((3,3,3))
            conv_im = convolve(skeleton,kernel)
            branch_image = (conv_im > 3) * skeleton

            # assume connected components are a single branch point
            all_labels = measure.label(branch_image)
            num_branchs = np.max(all_labels) 
                
            all_num_branches.append(num_branchs)
            
np.savetxt(f'./real_branch_stats.txt',np.array(all_num_branches))
    
