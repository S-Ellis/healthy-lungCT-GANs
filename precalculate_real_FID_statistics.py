#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Sam Ellis
Copyright 2022 Sam Ellis, King's College London
"""
import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import patch_dataset
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3

parser = argparse.ArgumentParser()
parser.add_argument('--patches_per_CT', type=int, default=20,help='Number of patches to load per CT image') # 20 patches per 509 CTs gives 10180 samples for FID calculation
parser.add_argument('--outf', default='real_samples/', help='Folder to output real images for FID calculation')
parser.add_argument('--cudaDevice', default='', help="cuda device number to use")

opt = parser.parse_args()
print(opt)

if opt.cudaDevice == '':
    device = torch.device("cpu")
else:
    device = torch.device("cuda:"+opt.cudaDevice)
#%% first, load the train filelists -----------------------------------------
# these should have been previously generated and will be in the current path
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
        
        data_full = (data_full + 1.) /2. # map from [-1,1] to [0,1]
        
        for jj in range(data_full.size(0)):
           vutils.save_image(torch.squeeze(data_full[jj, :, 16, :, :]), f'{outf}im_{i}_{jj}.png')
    
    
#%% when the folder is populated, calculate the activation stats and save
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]

model = InceptionV3([block_idx]).to(device)

mu, sigma = fid_score.compute_statistics_of_path(outf, model, batch_size=50,dims=2048, device=device)
                  
np.savez_compressed('./fid_real_stats.npz', mu=mu, sigma=sigma)
    
    
