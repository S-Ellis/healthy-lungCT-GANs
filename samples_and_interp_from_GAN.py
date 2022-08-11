#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Sam Ellis
Copyright 2022 Sam Ellis, King's College London
"""
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import generators as G
import utils
import ast

#%% parse args
parser = argparse.ArgumentParser()
parser.add_argument('--netG', default='', help="Path to netG")
parser.add_argument('--manualSeed', type=int, help='Manual seed')
parser.add_argument('--cudaDevice', default='', help="cuda device number to use")
parser.add_argument('--genModel', type=str,default='styleGAN', help="DCGAN, BigGAN or styleGAN")
parser.add_argument('--w_pass', type=str,default='False', help="Pass w directly to styleGAN?")
parser.add_argument('--interpOutput', type=str,default='collage', help="Output interps as a collage image or separately")
parser.add_argument('--n_interps', type=int, default=10, help='Number of intermediate interpolations to perform')
parser.add_argument('--samplesFlag', type=str,default='True', help="Flag to generate samples or skip to interps")

opt = parser.parse_args()
opt.w_pass = opt.w_pass == 'True'
opt.samplesFlag = opt.samplesFlag == 'True'
 
genName = opt.genModel

if (genName == 'DCGAN') or (genName == 'BigGAN'):
    opt.w_pass = False

#%% set up output folders
outf_interp = os.path.dirname(opt.netG) + '/samples_interp'
outf_samples = os.path.dirname(opt.netG) + '/samples/'

if opt.w_pass:
    outf_interp = outf_interp + '_w/'
else:
    outf_interp = outf_interp + '/'
    
try:
    os.makedirs(outf_interp)
    os.makedirs(outf_samples)
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

#%% 1) generate 64 2D images
if opt.samplesFlag:
    fake_all = []
    with torch.no_grad():
        z = torch.randn(64, nz, device=device)
        
        for ii in range(32):
            print(f'{ii} of 32')
            # generate 2 at a time
            fake = netG(z[2*ii:2*ii+2])
            fake = (fake + 1.) /2.
            
            fake_all.append(fake[0:1])
            fake_all.append(fake[1:2])
            
            for kk in range(fake.shape[0]):
                vutils.save_image(fake[kk, :, 16, :, :], f'{outf_samples}/sample_{ii}_{kk}.png',normalize=True)
    
    vutils.save_image(torch.cat(fake_all)[:, :, 16, :, :], f'{outf_samples}/samples_full.png',normalize=True)
    
#%% 2) interp between 2 images (lerp if passing w to styleGAN, slerp otherwise)
n_steps = 10 # the total number of interpolations to perform
with torch.no_grad():
    for ii in range(n_steps):
        print(f'\rStep {ii+1} of {n_steps}\n',end="")
        z_1 = torch.randn(1, nz, device=device)
        z_2 = torch.randn(1, nz, device=device)
        
        vals = np.linspace(0,1,opt.n_interps+2)
        
        if not opt.w_pass:
            z_noise_slerp = torch.zeros(opt.n_interps+2,nz,device=device)
            for jj in range(len(vals)):
                z_noise_slerp[jj] = utils.slerp(vals[jj],z_1,z_2)
            
            fake = torch.zeros(z_noise_slerp.shape[0],1,32,64,64)
            for zz in range(z_noise_slerp.shape[0]//2):
                print(f'{zz} of {z_noise_slerp.shape[0]//2} generations')
                fake[2*zz:2*zz+2] = netG(z_noise_slerp[2*zz:2*zz+2])
            
            fake = (fake + 1.) /2.

            if opt.interpOutput == 'collage':
                vutils.save_image(fake[:, :, 16, :, :], f'{outf_interp}interp_{ii}.png',normalize=True,nrow=n_steps+2)
            elif opt.interpOutput == 'separate':
                try:
                    os.makedirs(f'{outf_interp}/ims_{ii}/')
                except:
                    pass
                for kk in range(fake.shape[0]):
                    vutils.save_image(fake[kk, :, 16, :, :], f'{outf_interp}/ims_{ii}/im_{kk:04d}.png',normalize=True)

    
        else:
            
            # pass linearly interpolated when passing w directly
            w_both = netG.latentMapping(torch.squeeze(torch.cat((z_1,z_2))))
            w_1 = w_both[0:1]
            w_2 = w_both[1:2]
        
            w_noise_lerp = torch.zeros(opt.n_interps+2,nz,device=device)
            for jj in range(len(vals)):
                w_noise_lerp[jj] = torch.lerp(w_1,w_2,vals[jj])
                
            fake = torch.zeros(w_noise_lerp.shape[0],1,32,64,64)
            for ww in range(w_noise_lerp.shape[0]//2):
                print(f'{ww} of {w_noise_lerp.shape[0]//2} generations')
                fake[2*ww:2*ww+2] = netG(w_noise_lerp[2*ww:2*ww+2],w_passed=True)
                            
            fake = (fake + 1.) /2.
        
            if opt.interpOutput == 'collage':
                vutils.save_image(fake[:, :, 16, :, :], f'{outf_interp}interp_{ii}.png',normalize=True,nrow=n_steps+2)
            elif opt.interpOutput == 'separate':
                try:
                    os.makedirs(f'{outf_interp}/ims_{ii}/')
                except:
                    pass
                for kk in range(fake.shape[0]):
                    vutils.save_image(fake[kk, :, 16, :, :], f'{outf_interp}/ims_{ii}/im_{kk:04d}.png',normalize=True)
            