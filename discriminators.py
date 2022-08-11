#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Sam Ellis
Copyright 2022 Sam Ellis, King's College London
"""
import torch
import torch.nn as nn
from generators import Self_Attention

def distanceAndAppend_min(x_batch):
#    print('MDmin layer')
    batch_size = x_batch.shape[0]
    s = 1e6 * torch.ones(batch_size).to(x_batch.device)
    for ii in range(batch_size):
        for jj in range(batch_size):
            if ii != jj:
                s[ii] = min(s[ii],torch.abs(x_batch[ii] - x_batch[jj]).mean())
                
    s_full = s[:,None,None,None,None].repeat((1,1,2,4,4))

    return torch.cat((x_batch,s_full),dim=1), s

class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=1, SAflag = False, MDmin=False):
        super(Discriminator, self).__init__()
        self.SAflag = SAflag
        self.MDmin = MDmin
        self.block1 = nn.Sequential(
            nn.Conv3d(nc, ndf, (2,4,4), (2,2,2), (0,1,1), bias=False),
            nn.LeakyReLU(0.2, inplace=True))
            
        self.block2 = nn.Sequential(
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True))
        
        if (self.SAflag == True):
            self.SA = Self_Attention(ndf * 2)
            
        self.block3 = nn.Sequential(
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True))
            
        self.block4 = nn.Sequential(
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True))
            
        if (self.MDmin == True):
            self.block5 = nn.Sequential(
                nn.Conv3d(ndf * 8 + 1, 1, (2,4,4), 1, 0, bias=False))
        else:
            self.block5 = nn.Sequential(
                nn.Conv3d(ndf * 8, 1, (2,4,4), 1, 0, bias=False))

    def forward(self, input, mode='trainDis'):
        
        h = self.block1(input)
        h = self.block2(h)
        
        if (self.SAflag == True):
            h, attn = self.SA(h)
        
        h = self.block3(h)
        h = self.block4(h)
        
        if mode=='trainDis':
            if (self.MDmin == True):
                h, s = distanceAndAppend_min(h)
            output = self.block5(h)        
            return output.view(-1, 1).squeeze(1)
        elif mode=='returnFeats':
            return h
