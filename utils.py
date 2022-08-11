#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Sam Ellis
Copyright 2022 Sam Ellis, King's College London
"""
import torch
import discriminators as D
import numpy as np
import torchvision.utils as vutils
from pytorch_fid import fid_score
import shutil
import os
import generators as G

#%% function to sample most mode-collapsed samples
def selectSimilarSamples(outNum, inNum, nz, netG, netD):
    print(f'Selecting {outNum} most similar from {inNum} samples')
    with torch.no_grad():
        
        z = torch.randn(inNum, nz, device=next(netG.parameters()).device)
        
        # split into chunks of 32 for passing through, for memory purposes
        z_split = torch.split(z,32)
        all_feats = []
        for kk in range(len(z_split)):
        
            fake = netG(z_split[kk])
        
            feats = netD(fake, mode='returnFeats')
            
            all_feats.append(feats)
            
        all_feats = torch.cat(all_feats)
        
        _, similarities = D.distanceAndAppend_min(all_feats)
        
        _, idx = torch.sort(similarities)
        
        idx_out = idx[0:min(outNum,len(idx))]
        
        return z[idx_out], similarities
    
def getParams(model):
    a = list(model.parameters())
    b = [a[i].detach().cpu().numpy() for i in range(len(a))]
    c = [b[i].flatten() for i in range(len(b))]
    d = np.hstack(c)
    return d

def fid(netG, real_stats_filename, n_samples,device, outf, delete_samples=True):
    
    newOutf = f'{outf}/samples/'
    os.makedirs(newOutf)
    
    n_steps = n_samples // 50
    
    if ((netG.__class__) == G.DCGAN_gen) or ((netG.__class__) == G.bigGAN_gen):
        # according to https://discuss.pytorch.org/t/why-dont-we-put-models-in-train-or-eval-modes-in-dcgan-example/7422/3
        # when running some models in inference, you need to run some generations in train mode to stabilise BN statistics,
        # THEN go to eval mode
        print('Running some generations to stabilise BN stats...')
        for jj in range(20):
            z_noise = torch.randn(50, netG.nz, device=device)
            fake = netG(z_noise)
            
        netG.eval()
        if netG.training == False:
            print('netG in eval mode')
    else:
        netG.eval()
        if netG.training == False:
            print('netG in eval mode')
    
    # 1) generate the samples
    with torch.no_grad():
        for ii in range(n_steps):
            if ii == n_steps-1:
                batchSize = np.mod(n_samples,50)
            else:
                batchSize = 50
            print(f'\rStep {ii+1} of {n_steps}',end="")
            z_noise = torch.randn(batchSize, netG.nz, device=device)
                  
            fake = netG(z_noise)
            
            fake = (fake + 1.) /2. # map from [-1,1] to [0,1]
            
            for jj in range(fake.size(0)):
               vutils.save_image(torch.squeeze(fake[jj, :, 16, :, :]), f'{newOutf}im_{ii}_{jj}.png')
    
    print('')
    netG.train()
    if netG.training == True:
        print('netG in train mode')
    
    # 2) calculate the fid
    fid_value = fid_score.calculate_fid_given_paths(paths = [real_stats_filename, f'{newOutf}'],
                                          batch_size = 50,
                                          device = device,
                                          dims = 2048)

    # 3) clearup the folder
    if delete_samples:
        shutil.rmtree(f'{outf}/samples/')
    
    # 4) return the value
    return fid_value

def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res
    
    