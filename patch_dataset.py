#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Sam Ellis
Copyright 2022 Sam Ellis, King's College London
"""
import numpy as np
import os
from torch.utils.data import Dataset
import torch
import SimpleITK as sitk
from skimage.util import view_as_windows
import pylidc as pl # module for handling the LIDC dataset
from pylidc.utils import consensus

#%%
def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing

def worldToVoxelCoord(worldCoord, origin, spacing):
     
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord // spacing
    return [int(i) for i in voxelCoord]

def normalizeVolume(npzarray):
    npzarray = npzarray
    
    maxHU = 400.
    minHU = -1000.
 
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    
    npzarray = 2.0 * npzarray - 1.0
    return npzarray

#%% data loading
class patchLoader(Dataset):
    
    def __init__(self, imageListCT, lungSegListCT, num_patch_per_CT, dType=torch.float32):
        # imageListCT should be a list of FULL PATHS to mhd files
        # lungSegListCT should be a list of FULL PATHS to seg mhd files
        
        self.imageListCT = imageListCT
        self.lungSegListCT = lungSegListCT
        self.num_patch_per_CT = num_patch_per_CT
        self.dType = dType
        self.patch_size = [32,64,64]
                
    def __len__(self):
        return len(self.imageListCT)
    
    def __getitem__(self,idx):
        
        image_file = self.imageListCT[idx]
        series_uid_im = os.path.basename(image_file)[0:-4]
        lungCT, _, _ = load_itk_image(image_file)
        
        seg_file = self.lungSegListCT[idx]
        series_uid_seg = os.path.basename(image_file)[0:-4]        
        numpySeg, _, _ = load_itk_image(seg_file)
        lungMask = np.logical_or(numpySeg == 3, numpySeg == 4).astype('int16') # (for both lungs)
        
        # check that the series_uids match!
        assert series_uid_im==series_uid_seg
        
        series_uid = series_uid_im
        
        # get the nodule information from pylidc and generate a nodule mask image which is the same size as the original image
        scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == series_uid)[0]
        nods_pl = scan.cluster_annotations()
        nodMask = np.zeros_like(lungCT)
        nodMask_wide = nodMask.copy()
        for anns in nods_pl:
            cmask,cbbox,masks = consensus(anns, clevel=0.5, pad=[(0,0), (0,0), (0,0)])
            cmask = np.swapaxes(cmask,1,2);cmask = np.swapaxes(cmask,0,1)
            nodMask[cbbox[2].start:cbbox[2].stop,cbbox[0].start:cbbox[0].stop,cbbox[1].start:cbbox[1].stop] = cmask
            nodMask_wide[cbbox[2].start - self.patch_size[0]//2 - 1:cbbox[2].stop + self.patch_size[0]//2 + 1,
                         cbbox[0].start - self.patch_size[1]//2 - 1:cbbox[0].stop + self.patch_size[1]//2 + 1,
                         cbbox[1].start - self.patch_size[2]//2 - 1:cbbox[1].stop + self.patch_size[2]//2 + 1] = 1
                         

        # zero mask close to edges to avoid going out of bounds when choosing index
        selectionMask = lungMask.copy()
        selectionMask[:] = 0
        
        
        selectionMask[self.patch_size[0]//2:-self.patch_size[0]//2,
                      self.patch_size[1]//2:-self.patch_size[1]//2,
                      self.patch_size[2]//2:-self.patch_size[2]//2] = lungMask[self.patch_size[0]//2:-self.patch_size[0]//2,
                      self.patch_size[1]//2:-self.patch_size[1]//2,
                      self.patch_size[2]//2:-self.patch_size[2]//2]
        selectionMask[np.where(nodMask_wide==1)] = 0
        
        validIdx = np.stack(np.where(selectionMask==1))
                    
        sampled_idx = np.random.randint(0,validIdx.shape[1],self.num_patch_per_CT)
        
        patch_centres = validIdx[:,sampled_idx]
        
        # pad and get patch view and then extract
        lungCT_pad = np.pad(lungCT,((self.patch_size[0]//2,self.patch_size[0]//2-1),(self.patch_size[1]//2,self.patch_size[1]//2-1),(self.patch_size[2]//2,self.patch_size[2]//2-1)),mode='constant')
        patch_view = view_as_windows(lungCT_pad, self.patch_size)
        
        extractedPatches = patch_view[tuple(patch_centres)].copy()

        extractedPatches = torch.tensor(normalizeVolume(extractedPatches.astype('float32')))

        return extractedPatches[:,None,:,:,:]