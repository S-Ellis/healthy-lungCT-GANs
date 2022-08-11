#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Sam Ellis
Copyright 2022 Sam Ellis, King's College London
"""
import torch
import numpy as np
from pytorch_fid import fid_score
import os
import generators as G
from Med3D.setting import parse_opts
from Med3D.model import generate_model
import torch.nn as nn
import ast

def load_med3d_backbone():
    # setttings
    sets = parse_opts()
    sets.resume_path = './Med3D/pretrain/resnet_10_23dataset.pth'
    sets.target_type = "normal"
    sets.phase = 'test'
    sets.no_cuda = True
    sets.model_depth = 10
    sets.resnet_shortcut = 'B'

    # getting model
    checkpoint = torch.load(sets.resume_path)
    net, _ = generate_model(sets)
    net_dict = net.state_dict()

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = checkpoint['state_dict']

    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    
    net_dict.update(new_state_dict)
    net.load_state_dict(net_dict)

    net_backbone = nn.Sequential(*[list(net.children())[i] for i in range(8)]) # trim off the segmentation head

    return net_backbone

def normalise_image_med3d(in_batch):
    batch_size = in_batch.shape[0]
    channel_size = in_batch.shape[1]
    z = in_batch.shape[2]
    x = in_batch.shape[3]
    y = in_batch.shape[4]
    in_batch_flat = in_batch.reshape(batch_size,-1)
    mean = in_batch_flat.mean(dim=1)
    std = in_batch_flat.std(dim=1)
    
    mean = mean[:,None,None,None,None].repeat(1,channel_size,z,x,y)
    std = std[:,None,None,None,None].repeat(1,channel_size,z,x,y)

    out = (in_batch - mean)/std
    
    return out


def stats_from_net(netG, n_samples, device, med3d_net, channels_to_use = -1, globalAvg=True):
           
    n_steps = n_samples // 50
    
    if ((netG.__class__) == G.DCGAN_gen) or ((netG.__class__) == G.bigGAN_gen):
        # according to https://discuss.pytorch.org/t/why-dont-we-put-models-in-train-or-eval-modes-in-dcgan-example/7422/3
        # when running DCGAN in inference, you need to run some generations in train mode to stabilise BN statistics,
        # THEN go to eval mode
        print('Running some generations to stabilise BN stats...')
        for jj in range(20):
            with torch.no_grad():
                z_noise = torch.randn(100, netG.nz, device=device)
                fake = netG(z_noise)
            
        netG.eval()
        if netG.training == False:
            print('netG in eval mode')
    else:
        netG.eval()
        if netG.training == False:
            print('netG in eval mode')
    
    act_arr = []
    with torch.no_grad():
        for ii in range(n_steps):
            
            if ii == n_steps-1:
                batchSize = np.mod(n_samples,50)
            else:
                batchSize = 50
                
            batchSize = np.max([batchSize,2]) 
            print(f'\rStep {ii+1} of {n_steps}',end="")
            z_noise = torch.randn(batchSize, netG.nz, device=device)
            
            fake = netG(z_noise)

            fake = (fake + 1.) /2.
            
            tmp_act = med3d_net(normalise_image_med3d(fake))
            
            if channels_to_use == -1:
                channels_to_use = tmp_act.shape[1]
            
            if globalAvg == True:
                act_arr.append(tmp_act[:,0:channels_to_use].mean(dim=(2,3,4)))
            else:
                act_arr.append(tmp_act[:,0:channels_to_use])

        act_arr = torch.cat(act_arr)
        num_images = act_arr.shape[0]
        act_arr = act_arr.reshape(num_images,-1)
    
    mu = np.mean(act_arr.cpu().numpy(), axis=0)
    
    sigma = np.cov(act_arr.cpu().numpy().astype('float16'), rowvar=False)

    return mu, sigma

def stats_from_real_folder_3d(real_path, device, med3d_net, channels_to_use = -1, globalAvg=True):
    files = os.listdir(real_path)
    
    act_arr = []
    with torch.no_grad():
    
        for ii, file in enumerate(files):
            curr_arr = torch.load(real_path + file).to(device)
            tmp_act = med3d_net(normalise_image_med3d(curr_arr))
            
            if channels_to_use == -1:
                channels_to_use = tmp_act.shape[1]
            
            if globalAvg == True:
                act_arr.append(tmp_act[:,0:channels_to_use].mean(dim=(2,3,4)))
            else:
                act_arr.append(tmp_act[:,0:channels_to_use])
            
        act_arr = torch.cat(act_arr)
        num_images = act_arr.shape[0]
        act_arr = act_arr.reshape(num_images,-1)
            
    mu = np.mean(act_arr.cpu().numpy(), axis=0)
    
    sigma = np.cov(act_arr.cpu().numpy().astype('float16'), rowvar=False)

    return mu, sigma
    

def fid3d(real_folder, netG, n_samples, device):
    
    med3d_net = load_med3d_backbone().to(device)
    
    mu2, sigma2 = stats_from_net(netG, n_samples, device, med3d_net,channels_to_use=-1)
    print(mu2.shape,sigma2.shape)

    mu1, sigma1 = stats_from_real_folder_3d(real_folder, device, med3d_net,channels_to_use=-1)
    print(mu1.shape,sigma1.shape)

    fid_val = fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    return fid_val

#%% main script
import argparse
parser_main = argparse.ArgumentParser()
parser_main.add_argument('--netG', default='', help="Path to netG")
parser_main.add_argument('--cudaDevice', default='', help="cuda device number to use")
parser_main.add_argument('--genModel', type=str,default='styleGAN', help="DCGAN, styleGAN, or BigGAN")
parser_main.add_argument('--fid_samples', type=int, default=10180, help='Number of samples to generate to calculate FID')
opt = parser_main.parse_known_args()[0]

if opt.cudaDevice == '':
    device = torch.device("cpu")
else:
    device = torch.device("cuda:"+opt.cudaDevice)

genName = opt.genModel
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
    file = open(f'{path_only}/biggan_args.txt', 'r')
    contents = file.read()
    config = ast.literal_eval(contents)
    netG = G.bigGAN_gen(**config).to(device)

netG.load_state_dict(torch.load(opt.netG,map_location=device))
netG = netG.to(device)

real_path = './real_samples_3D_FID/'

fid_val = fid3d(real_path, netG, opt.fid_samples, device)
print('#####################')
print(fid_val)
print('#####################')
      
np.savetxt(f'{path_only}/FID_3d.txt',np.atleast_1d(np.asarray(fid_val)))




