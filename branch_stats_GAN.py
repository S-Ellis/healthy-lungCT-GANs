#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Sam Ellis
Copyright 2022 Sam Ellis, King's College London
"""
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from skimage.transform import rescale
from scipy.ndimage import convolve
from skimage import measure
import generators as G
import random
import ast

#%% parse args
parser = argparse.ArgumentParser()
parser.add_argument('--netG', default='', help="Path to netG")
parser.add_argument('--manualSeed', type=int, help='Manual seed')
parser.add_argument('--batchSize', type=int, default=20, help='Input batch size')
parser.add_argument('--num_batches', type=int, default=509, help='Number of batches')
parser.add_argument('--cudaDevice', default='', help="cuda device number to use")
parser.add_argument('--genModel', type=str,default='styleGAN', help="DCGAN, BigGAN or styleGAN")

opt = parser.parse_args() 
genName = opt.genModel

#%% set up output folders
outf = os.path.dirname(opt.netG) + '/branch_stats_3D/'
try:
    os.makedirs(outf)
except OSError:
    pass

if opt.cudaDevice == '':
    device = torch.device("cpu")
else:
    device = torch.device("cuda:"+opt.cudaDevice)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)

cudnn.benchmark = True

#%% load the generator model:
path_only = os.path.split(opt.netG)[0]

file = open(f'{path_only}/args.txt', 'r')
contents = file.read()
args = ast.literal_eval(contents)
nz = args['nz']

if genName == 'DCGAN':
    netG = G.DCGAN_gen(nz).to(device)
elif genName == 'styleGAN':
    netG = G.styleGAN_gen(nz, styleMixing=False, mappingNet=args['mappingNet']).to(device)
elif genName == 'BigGAN':
    # load the biggan archi options
    file = open(f'{path_only}/biggan_args.txt', 'r')
    contents = file.read()
    config = ast.literal_eval(contents)
    netG = G.bigGAN_gen(**config).to(device)
    
netG.load_state_dict(torch.load(opt.netG,map_location=device))
netG = netG.to(device)

if (genName == 'DCGAN') or (genName == 'BigGAN'):
    print('Running some generations to stabilise BN stats...')
    with torch.no_grad():
        for jj in range(100):
            print(f'{jj} of 100')
            z_noise = torch.randn(2, netG.nz, device=device)
            fake = netG(z_noise)
netG.eval()

#%% perform the sampling
all_num_branches = []
with torch.no_grad():
    for ii in range(opt.num_batches):
        print(f'{ii} of {opt.num_batches}')
        z = torch.randn(opt.batchSize, nz, device=device)
        
        fake = netG(z)
        fake = (fake + 1.) /2.
                
        # loop over each image in the batch
        for jj in range(fake.size(0)):
            curr_im = np.squeeze(fake[jj, :, :, :, :].cpu().numpy())
            
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
        
# output to file
np.savetxt(f'{outf}/branch_stats.txt',np.array(all_num_branches))
    
