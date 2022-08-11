#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Sam Ellis
Copyright 2022 Sam Ellis, King's College London
"""
import argparse
import os
import time
import random
import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import numpy as np
import patch_dataset
import generators as G
import discriminators as D
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='Number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=48, help='Input batch size')
parser.add_argument('--nz', type=int, default=512, help='Size of the latent z vector')
parser.add_argument('--niter', type=int, default=20, help='Number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate, default=0.0001')
parser.add_argument('--patches_per_CT', type=int, default=672, help='Number of patches to load per CT image')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--netG', default='', help="Path to netG (to continue training)")
parser.add_argument('--netD', default='', help="Path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='Manual seed')
parser.add_argument('--cudaDevice', default='', help="cuda device number to use")
parser.add_argument('--MDmin', type=str,default='True', help="Use MDmin or not")
parser.add_argument('--genModel', type=str,default='styleGAN', help="DCGAN, styleGAN or BigGAN")
parser.add_argument('--loss', type=str,default='relativistic', help="Standard or relativistic loss")
parser.add_argument('--mappingNet', type=str,default='True', help="Use styleGAN mapping network")
parser.add_argument('--styleMixing', type=str,default='True', help="Use styleGAN style mixing regularisation")
parser.add_argument('--ep_switch', type=int, default=0, help='Number of epochs to run before largeEBS takes effect')
parser.add_argument('--largeEBS', type=int, default=0, help='Effective batch size when using the largeEBS technique')
parser.add_argument('--fidFlag', type=str,default='True', help="Calculate FID through training")
parser.add_argument('--fid_samples', type=int, default=10180, help='Number of samples to generate to calculate FID')

#%% parse args ------------------------------------------------------------
[opt,_] = parser.parse_known_args()
opt.MDmin = opt.MDmin == 'True'
opt.mappingNet = opt.mappingNet == 'True'
opt.styleMixing = opt.styleMixing == 'True'
opt.fidFlag = opt.fidFlag == 'True'
opt.nz = int(opt.nz)

#%% ascertain the output directory and make -------------------------------
if (opt.largeEBS > opt.batchSize):
    largeEBS_dir_name = f'_largeEBS_{opt.largeEBS}'
else:
    largeEBS_dir_name = ''

genName = opt.genModel

if opt.MDmin:
    disName = '_MDmin'
else:
    disName = ''
    
if (not opt.mappingNet) and (opt.genModel == 'styleGAN'):
    mappingNetName = '_noMapping'
else:
    mappingNetName = ''
    
if not (opt.styleMixing) and (opt.genModel == 'styleGAN'):
    styleMixingName = '_noStyleMixing'
else:
    styleMixingName = ''
    
outf = './' + genName +  disName + '_' + opt.loss + largeEBS_dir_name + mappingNetName + styleMixingName + '/'

if opt.cudaDevice == '':
    device = torch.device("cpu")
else:
    device = torch.device("cuda:"+opt.cudaDevice)

try:
    outf = outf + f'/{time.time()}'
    os.makedirs(outf)
except OSError:
    pass

# output the final opts to a file
print(vars(opt), file=open(f'{outf}/args.txt', 'w'))

#%% manual seeds -----------------------------------------------------------
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)

cudnn.benchmark = True

assert np.mod(opt.patches_per_CT, opt.batchSize) == 0
batches_per_CT = opt.patches_per_CT // opt.batchSize 
        
#%% first, load the train filelists -----------------------------------------
# these should have been previously generated and will be in the current path
lungSegListCT = np.atleast_1d(np.loadtxt('lungSegList_train.txt','str'))
        
imageListCT = np.atleast_1d(np.loadtxt('imageList_train.txt','str'))                
                
# sanity check, the two vectors should be the same filenames
if not [os.path.basename(lungSegListCT[ii]) for ii in range(len(lungSegListCT))] == [os.path.basename(imageListCT[ii]) for ii in range(len(imageListCT))]:
    raise ValueError('The file lists do not match')
    
dataset = patch_dataset.patchLoader(imageListCT, lungSegListCT, num_patch_per_CT = opt.patches_per_CT)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True, num_workers=int(opt.workers)) # load in patches from a single image at a time
    
#%% setup the generator model:
if genName == 'DCGAN':
    netG = G.DCGAN_gen(opt.nz).to(device)
elif genName == 'styleGAN':
    netG = G.styleGAN_gen(opt.nz, styleMixing=opt.styleMixing, mappingNet=opt.mappingNet).to(device)
elif genName == 'BigGAN':
    # use most settings as default
    import biggan_parser
    parser2 = biggan_parser.prepare_parser()
    [config, _] = parser2.parse_known_args()
    config = vars(config)
    config['dim_z'] = opt.nz
    config['resolution'] = 64

    print(config, file=open(f'{outf}/biggan_args.txt', 'w'))
    netG = G.bigGAN_gen(**config).to(device)
        
if opt.netG!='':
    netG.load_state_dict((torch.load(opt.netG,map_location=device)))

#%% setup the discriminator model
netD = D.Discriminator(MDmin = opt.MDmin).to(device)

if opt.netD!='':
    netD.load_state_dict((torch.load(opt.netD,map_location=device)))
#%% load the losses
if opt.loss == 'standard':
    import losses_standard as L    
elif opt.loss == 'relativistic':
    import losses_relativistic as L
    
#%%
fixed_noise = torch.randn(opt.batchSize, opt.nz, device=device)

# setup optimizers
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

#%%
allErrG = []
allErrD = []
allFID = []
allLabels = []

bestFID = 1.e6
    
for epoch in range(opt.niter):

    for i, data_full in enumerate(dataloader, 0):
        
        errD_patient = 0.
        errG_patient = 0.
        
        # make a shuffled version so that G and D see different mini-batches of real data
        data_full_shuffle = data_full[0][torch.randperm(data_full.shape[1])] 
        data_full_shuffle_split = torch.split(data_full_shuffle,opt.batchSize)
        
        data_full_split = torch.split(data_full[0],opt.batchSize)
        
        for jj in range(len(data_full_split)):
            
            ############################
            # (1) Update D network:
            ###########################
            # real batch
            netD.zero_grad()
            real_cpu = data_full_split[jj].to(device)
            batch_size = real_cpu.size(0)
            critic_real = netD(real_cpu)
    
            # fake batch
            if (epoch >= opt.ep_switch)  and (opt.largeEBS > opt.batchSize):
                print('Preselecting similar samples')
                noise,similarities = utils.selectSimilarSamples(outNum=batch_size, inNum=opt.largeEBS, nz=opt.nz, netG=netG, netD=netD)
            else:
                noise = torch.randn(batch_size, opt.nz, device=device)
            fake = netG(noise)
            critic_fake = netD(fake.detach())
            
            # calculate the loss function and send backwards            
            errD = L.loss_dis(critic_real,critic_fake)
            errD.backward()
            optimizerD.step()
            errD_patient += errD.item() / len(torch.split(data_full[0],opt.batchSize))

            
            ############################
            # (2) Update G network
            ###########################
            
            # real batch
            netG.zero_grad()
            real_cpu = data_full_shuffle_split[jj].to(device)
            batch_size = real_cpu.size(0)
            critic_real = netD(real_cpu)
            
            # fake batch
            if (epoch >= opt.ep_switch)  and (opt.largeEBS > opt.batchSize):
                print('Preselecting similar samples')
                noise,similarities = utils.selectSimilarSamples(outNum=batch_size, inNum=opt.largeEBS, nz=opt.nz, netG=netG, netD=netD)
            else:
                noise = torch.randn(batch_size, opt.nz, device=device)
            fake = netG(noise)
            critic_fake = netD(fake)
            
            # losses
            errG = L.loss_gen(critic_real,critic_fake)
                
            errG.backward()
            optimizerG.step()

            errG_patient += errG.item() / len(torch.split(data_full[0],opt.batchSize))

            #%============ output and FID calculation
            print('[%d/%d][%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch, opt.niter, i, len(dataloader), jj, len(torch.split(data_full[0],opt.batchSize)),  
                     errD.item(), errG.item()))
            
            if ((i % 100 == 0) | (i == (len(dataloader)-1))) & ((jj == 0) | (jj == 1)): 
                real_cpu_save = real_cpu[:,:,16].clone()

                vutils.save_image(real_cpu_save,
                        '%s/real_samples_%03d.png' % (outf, jj), # this is overwritten every epoch, that's fine
                        normalize=True)
                fake = netG(fixed_noise)
                fake = fake[:,:,16]
                
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d_%03d_%03d.png' % (outf, epoch, i, jj),
                        normalize=True)
            
            if ((i == (len(dataloader)-1)) | (i % 200 == 0)) & (jj == 0):
                # do checkpointing at end of epoch
                torch.save(netG.state_dict(), '%s/netG_curr.pth' % (outf))
                torch.save(netD.state_dict(), '%s/netD_curr.pth' % (outf))
                
                np.savetxt(f'{outf}/last_epoch.txt',np.asarray([epoch,i]))
                
                # caclulate FID too
                if opt.fidFlag:
                    FID = utils.fid(netG,'./fid_real_stats.npz',n_samples=opt.fid_samples,device=device,outf=outf)
                    allFID.append(FID)
                    allLabels.append(f'epoch{epoch}_{i}')
                    
                    np.savetxt(f'{outf}/allFID.txt',np.asarray(allFID))
                    np.savetxt(f'{outf}/allLabels.txt',np.asarray(allLabels),'%s')
                    
                    if FID < bestFID:
                        print(f'Best FID found:{bestFID} -> {FID}')
                        bestFID = FID
                        torch.save(netG.state_dict(), '%s/netG_best.pth' % (outf))
                        torch.save(netD.state_dict(), '%s/netD_best.pth' % (outf))
                        np.savetxt(f'{outf}/bestFID.txt',np.atleast_1d(np.asarray(FID)))
                        np.savetxt(f'{outf}/bestFID_epoch.txt',np.asarray([epoch,i]))
            
        # save progress at end of patient
        allErrG.append(errG_patient)
        allErrD.append(errD_patient)
        np.savetxt(f'{outf}/allErrG.txt',np.asarray(allErrG))
        np.savetxt(f'{outf}/allErrD.txt',np.asarray(allErrD))
