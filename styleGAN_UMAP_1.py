#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Sam Ellis
Copyright 2022 Sam Ellis, King's College London
"""
import numpy as np
import torch
import generators as G
import os
import argparse
import ast
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from skimage.transform import rescale
from scipy.ndimage import convolve
from skimage import measure

#%% deal with args
parser = argparse.ArgumentParser()
parser.add_argument('--netG', default='', help="Path to netG (must be a styleGAN model that has the latent mapping network)")
parser.add_argument('--n_background', type=int, default=50000, help='Number of background samples to calculate UMAP embedding (large number required for stability in 512 dims)')
parser.add_argument('--n_examples', type=int, default=1000, help='Number of samples to generate images for')
parser.add_argument('--cudaDevice', default='', help="cuda device number to use")
opt = parser.parse_args()

outf = os.path.dirname(opt.netG) + '/UMAP_visualisation/'
try:
    os.makedirs(outf)
except OSError:
    pass

if opt.cudaDevice == '':
    device = torch.device("cpu")
else:
    device = torch.device("cuda:"+opt.cudaDevice)

#%% load our styleGAN
path_only = os.path.split(opt.netG)[0]

file = open(f'{path_only}/args.txt', 'r')
contents = file.read()
args = ast.literal_eval(contents)
nz = args['nz']

netG = G.styleGAN_gen(nz, styleMixing=False, mappingNet=True).to(device)
netG.load_state_dict(torch.load(opt.netG,map_location=device))

#%% sample from z and get w
z_background = torch.randn(opt.n_background,nz).to(device)
z_examples = torch.randn(opt.n_examples,nz).to(device)

#print('Mapping background z to w')
#w_background = []
#with torch.no_grad():
#    for ii, z_bg in enumerate(torch.split(z_background,opt.n_background//128)):
#        print(f'\rMapping batch {ii} of {len(torch.split(z_background,opt.n_background//128))} z to w',end="")
#        w_background.append(netG.latentMapping(z_bg).to(torch.device('cpu')))
#        
#w_background = torch.cat(w_background)
#print('')

print('Mapping z to w')
with torch.no_grad():
    w_background = netG.latentMapping(z_background)
    w_examples = netG.latentMapping(z_examples)

#%% pass through our examples that we want images for
fake_ims_examples = []
print('Generating images for examples')
with torch.no_grad():
    for kk, w_eg in enumerate(torch.split(w_examples,opt.n_examples//32)):
        print(f'\rGenerating images for batch {kk} of {len(torch.split(w_examples,opt.n_examples//32))} examples',end="")
        fake_ims_examples.append(netG(w_eg,w_passed=True))

fake_ims_examples = torch.cat(fake_ims_examples)
print('')

#%% calculate number of branch points per image
all_num_branches = []
for jj in range(fake_ims_examples.size(0)):
    print(f'\rCalculating branch points for {jj} of {fake_ims_examples.size(0)} examples',end="")

    curr_im = np.squeeze(fake_ims_examples[jj, :, :, :, :].cpu().numpy())
    
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

print('')

#%% save the outputs
np.save(f'{outf}/branch_stats_examples.npy',np.array(all_num_branches))
np.save(f'{outf}w_vectors_background.npy',w_background.cpu())
np.save(f'{outf}w_vectors_examples.npy',w_examples.cpu())
