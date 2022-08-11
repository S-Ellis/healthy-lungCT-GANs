#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Sam Ellis
Copyright 2022 Sam Ellis, King's College London
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def meanAndStd(f, eps=1e-5):
    batch_size, num_channels = f.shape[:2]
    
    # calculate the standard deviation across all elements per instance per channel
    feat_var = f.view(batch_size, num_channels, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(batch_size, num_channels, 1, 1, 1) #re-expand tp the original size
    
    # calculate the mean of the feature maps per instance per channel
    feat_mean = f.view(batch_size, num_channels, -1).mean(dim=2).view(batch_size, num_channels, 1, 1 ,1)
    return feat_mean, feat_std

def AdaIN(f_content, f_style):
    
    # get the shape of the input content feature map
    f_content_shape = f_content.size()
    nc = f_content_shape[1]
    
    # calculate the instance mean and standard deviation for content
    content_mean, content_std = meanAndStd(f_content)
    
    # split the f_style into the mean and std
    style_mean = f_style[:,0:nc][:,:,None,None,None].repeat((1,1,f_content_shape[2],f_content_shape[3],f_content_shape[4]))
    style_std = f_style[:,nc::][:,:,None,None,None].repeat((1,1,f_content_shape[2],f_content_shape[3],f_content_shape[4]))

    # normallise the content features
    f_content_norm = (f_content - content_mean.expand(f_content_shape)) / content_std.expand(f_content_shape)
    
    
    return f_content_norm * style_std + style_mean

def conditionalSplit(w,swapPoint,layerCtr,alreadySplit):
    
    if (layerCtr==swapPoint) & (alreadySplit!=True):
        
        w = w[torch.randperm(w.shape[0])] # shuffle along the batch dimension
        
    return w

class Self_Attention(nn.Module):
    def __init__(self,nc_in):
        super(Self_Attention,self).__init__()
        self.CBar = nc_in//8
        
        self.Wf = nn.Conv3d(in_channels = nc_in , out_channels = self.CBar , kernel_size= 1)
        self.Wg = nn.Conv3d(in_channels = nc_in , out_channels = self.CBar , kernel_size= 1)
        self.Wh = nn.Conv3d(in_channels = nc_in , out_channels = self.CBar , kernel_size= 1)
        self.Wv = nn.Conv3d(in_channels = nc_in//8 , out_channels = nc_in , kernel_size= 1)

        self.gamma = nn.Parameter(torch.zeros(1))        

        self.softmax  = nn.Softmax(dim=-1) 
    def forward(self,x,singleOutput=True):
#        print('Self attention!')
        batch_size, C, W ,H, D = x.size()
        f  = self.Wf(x).view(batch_size,-1,W*H*D)
        g =  self.Wg(x).view(batch_size,-1,W*H*D)
        s =  torch.bmm(f.permute(0,2,1),g)
        bta = self.softmax(s) 
        h = self.Wh(x).view(batch_size,-1,W*H*D)

        out = self.Wv(torch.bmm(h,bta.permute(0,2,1)).view(batch_size,self.CBar,W ,H, D))
        
        out = self.gamma*out + x
        
        if singleOutput == True:
            return out
        else:
            return out, bta
        

class styleGAN_gen(nn.Module):
    def __init__(self, nz, styleMixing=True, mappingNet=True, SAflag=False):
        super(styleGAN_gen, self).__init__()
        
        self.nz = nz
        self.styleMixing = styleMixing
        self.SAflag = SAflag
        self.mappingNet = mappingNet

        if self.mappingNet==True:
            self.latentMapping = nn.Sequential(
                    nn.Linear(nz,nz),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(nz,nz),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(nz,nz),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(nz,nz),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(nz,nz),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(nz,nz),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(nz,nz),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(nz,nz),
                    nn.LeakyReLU(0.2, inplace=True)
                )
        
        self.C1 = nn.Sequential(
            nn.Conv3d(512, 512, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        self.C2 = nn.Sequential(
            nn.Conv3d(512, 256, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        self.C3 = nn.Sequential(
            nn.Conv3d(256, 256, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        self.C4 = nn.Sequential(
            nn.Conv3d(256, 128, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        self.C5 = nn.Sequential(
            nn.Conv3d(128, 128, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        self.C6 = nn.Sequential(
            nn.Conv3d(128, 64, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        self.C7 = nn.Sequential(
            nn.Conv3d(64, 64, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        if (self.SAflag == True):
            self.SA = Self_Attention(64)
        
        self.C8 = nn.Sequential(
            nn.Conv3d(64, 32, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))

        self.C9 = nn.Sequential(
            nn.Conv3d(32, 32, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        self.C10 = nn.Sequential(
            nn.Conv3d(32, 16, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        self.C_out = nn.Sequential(
            nn.Conv3d(16, 1, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.Tanh())
        
        # affine mappings produce style stds and means (so, need 2x the output channels)
        self.A1 = nn.Linear(512,2*512)
        self.A2 = nn.Linear(512,2*512)
        self.A3 = nn.Linear(512,2*256)
        self.A4 = nn.Linear(512,2*256)
        self.A5 = nn.Linear(512,2*128)
        self.A6 = nn.Linear(512,2*128)
        self.A7 = nn.Linear(512,2*64)
        self.A8 = nn.Linear(512,2*64)
        self.A9 = nn.Linear(512,2*32)
        self.A10 = nn.Linear(512,2*32)
        self.A11 = nn.Linear(512,2*16)

    def forward(self, z_in, w_passed=False):
        batch_size = z_in.shape[0]
        num_channels = z_in.shape[1]
        assert num_channels == 512
        
        if (self.styleMixing == True) & (self.training == True):
            swapPoint = torch.randint(6,(1,1,1)).detach().item()
            alreadySplit = False
            layerCtr = 0
        
        if (self.mappingNet == False) or (w_passed == True):
            w = torch.squeeze(z_in)
        else: # if mapping network is in use and w_passed is false, use the mapping network
            w = self.latentMapping(torch.squeeze(z_in))
        
        constIm = torch.ones((batch_size,num_channels,1,2,2)).to(z_in.dtype).to(z_in.device)
        
        h = AdaIN(constIm,self.A1(w))        
       
        h = self.C1(h)
        
        if (self.styleMixing == True) & (self.training == True):
            w = conditionalSplit(w,swapPoint,layerCtr,alreadySplit)
            layerCtr += 1
        
        h = AdaIN(h,self.A2(w))        
        
        h = F.interpolate(h,scale_factor=2)
        
        h = self.C2(h)
        
        h = AdaIN(h,self.A3(w))  
        
        h = self.C3(h)

        if (self.styleMixing == True) & (self.training == True):
            w = conditionalSplit(w,swapPoint,layerCtr,alreadySplit)
            layerCtr += 1
        
        h = AdaIN(h,self.A4(w))  

        h = F.interpolate(h,scale_factor=2)

        h = self.C4(h)

        h = AdaIN(h,self.A5(w))  
        
        h = self.C5(h)

        if (self.styleMixing == True) & (self.training == True):
            w = conditionalSplit(w,swapPoint,layerCtr,alreadySplit)
            layerCtr += 1
        
        h = AdaIN(h,self.A6(w))  

        h = F.interpolate(h,scale_factor=2)
 
        h = self.C6(h)

        h = AdaIN(h,self.A7(w))  

        h = self.C7(h)
        
        if (self.SAflag == True):
            h, attn = self.SA(h)

        if (self.styleMixing == True) & (self.training == True):
            w = conditionalSplit(w,swapPoint,layerCtr,alreadySplit)
            layerCtr += 1
        
        h = AdaIN(h,self.A8(w)) 
        
        h = F.interpolate(h,scale_factor=2)
        
        h = self.C8(h)
        
        h = AdaIN(h,self.A9(w))  
        
        h = self.C9(h)
        
        if (self.styleMixing == True) & (self.training == True):
            w = conditionalSplit(w,swapPoint,layerCtr,alreadySplit)
            layerCtr += 1
        
        h = AdaIN(h,self.A10(w)) 
        
        h = F.interpolate(h,scale_factor=2)
        
        h = self.C10(h)
        
        if (self.styleMixing == True) & (self.training == True):
            w = conditionalSplit(w,swapPoint,layerCtr,alreadySplit)
            layerCtr += 1
        
        h = AdaIN(h,self.A11(w))  
        
        h = self.C_out(h)
        
        return h
    
class DCGAN_gen(nn.Module):
    def __init__(self, nz, ngf = 64, nc = 1):
        super(DCGAN_gen, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.ConvTranspose3d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 8, ngf * 4, (2,4,4), (2,2,2), (2,1,1), bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose3d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, h_in):
        
        h_in = h_in[:,:,None,None,None]

        output = self.main(h_in)
        return output
    
#############################################################################
# BigGAN generator architecture, minimally adapted from Brock et al's pytorch implementation
# https://github.com/ajbrock/BigGAN-PyTorch

import torch.nn as nn
import functools
from torch.nn import Parameter as P
from torch.nn import init
        
    
def biggan_G_arch3D(ch=64, attention='64', ksize='333333', dilation='111111'):
  arch = {}
  arch[64]  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [8, 16, 32, 64],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,7)}}
  arch[32]  = {'in_channels' :  [ch * item for item in [4, 4, 4]],
               'out_channels' : [ch * item for item in [4, 4, 4]],
               'upsample' : [True] * 3,
               'resolution' : [8, 16, 32],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,6)}}

  return arch

# Projection of x onto y
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())

# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x

# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs

class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))
  
  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values; 
  # note that these buffers are just for logging and are not used in training. 
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
   
  # Compute the spectrally-normalized weight
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv     
    return self.weight / svs[0]

# 3D Conv layer with spectral norm
class SNConv3d(nn.Conv3d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True, 
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
  def forward(self, x):
    return F.conv3d(x, self.W_(), self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)
    
# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Linear.__init__(self, in_features, out_features, bias)
    SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)

# Simple function to handle groupnorm norm stylization                      
def groupnorm(x, norm_style):
  # If number of channels specified in norm_style:
  if 'ch' in norm_style:
    ch = int(norm_style.split('_')[-1])
    groups = max(int(x.shape[1]) // ch, 1)
  # If number of groups specified in norm style
  elif 'grp' in norm_style:
    groups = int(norm_style.split('_')[-1])
  # If neither, default to groups = 16
  else:
    groups = 16
  return F.group_norm(x, groups)

class ccbn3d(nn.Module):
  def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1,
               cross_replica=False, mybn=False, norm_style='bn',):
    super(ccbn3d, self).__init__()
    self.output_size, self.input_size = output_size, input_size
    # Prepare gain and bias layers
    self.gain = which_linear(input_size, output_size)
    self.bias = which_linear(input_size, output_size)
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Use cross-replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    # Norm style?
    self.norm_style = norm_style
    
    if self.cross_replica:
        # not using this option so error if requested
        assert 1==2
#      self.bn = SyncBN3d(output_size, eps=self.eps, momentum=self.momentum, affine=False)
    elif self.mybn:
        # not using this option so error if requested
        assert 1==2
#      self.bn = myBN(output_size, self.eps, self.momentum)
    elif self.norm_style in ['bn', 'in']:
      self.register_buffer('stored_mean', torch.zeros(output_size))
      self.register_buffer('stored_var',  torch.ones(output_size)) 
    
    
  def forward(self, x, y):
    # Calculate class-conditional gains and biases
    gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1, 1)
    bias = self.bias(y).view(y.size(0), -1, 1, 1, 1)
    # If using my batchnorm
    if self.mybn or self.cross_replica:
      return self.bn(x, gain=gain, bias=bias)
    # else:
    else:
      if self.norm_style == 'bn':
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
      elif self.norm_style == 'in':
        out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
      elif self.norm_style == 'gn':
        out = groupnorm(x, self.normstyle)
      elif self.norm_style == 'nonorm':
        out = x
      return out * gain + bias
  def extra_repr(self):
    s = 'out: {output_size}, in: {input_size},'
    s +=' cross_replica={cross_replica}'
    return s.format(**self.__dict__)

# Convenience passthrough function
class identity(nn.Module):
  def forward(self, input):
    return input

# Generator blocks
# Note that this class assumes the kernel size and padding (and any other
# settings) have been selected in the main generator module and passed in
# through the which_conv arg. Similar rules apply with which_bn (the input
# size [which is actually the number of channels of the conditional info] must 
# be preselected)
class GBlock(nn.Module):
  def __init__(self, in_channels, out_channels,which_conv=nn.Conv2d, which_bn=ccbn3d, activation=None, 
               upsample=None):
    super(GBlock, self).__init__()
    
    self.in_channels, self.out_channels = in_channels, out_channels
    self.which_conv, self.which_bn = which_conv, which_bn
    self.activation = activation
    self.upsample = upsample
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.out_channels)
    self.conv2 = self.which_conv(self.out_channels, self.out_channels)
    self.learnable_sc = in_channels != out_channels or upsample
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels, 
                                     kernel_size=1, padding=0)
    # Batchnorm layers
    self.bn1 = self.which_bn(in_channels)
    self.bn2 = self.which_bn(out_channels)
    # upsample layers
    self.upsample = upsample

  def forward(self, x, y):
    h = self.activation(self.bn1(x, y))
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    h = self.conv1(h)
    h = self.activation(self.bn2(h, y))
    h = self.conv2(h)
    if self.learnable_sc:       
      x = self.conv_sc(x)
    return h + x

# reimplementation of SA for 3D
class Attention3d(nn.Module):
    def __init__(self,nc_in, which_conv=SNConv3d, name='attention'):
        super(Attention3d,self).__init__()
        
        self.which_conv = which_conv
        
        self.CBar = nc_in//8
        
        self.Wf = self.which_conv(in_channels = nc_in , out_channels = self.CBar , kernel_size= 1, padding=0, bias=False)
        self.Wg = self.which_conv(in_channels = nc_in , out_channels = self.CBar , kernel_size= 1, padding=0, bias=False)
        self.Wh = self.which_conv(in_channels = nc_in , out_channels = self.CBar , kernel_size= 1, padding=0, bias=False)
        self.Wv = self.which_conv(in_channels = nc_in//8 , out_channels = nc_in , kernel_size= 1, padding=0, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))        

        self.softmax  = nn.Softmax(dim=-1) 
    def forward(self,x,y=None):
#        print('Self attention!')
        batch_size, C, W ,H, D = x.size()
        f  = self.Wf(x).view(batch_size,-1,W*H*D)
        g =  self.Wg(x).view(batch_size,-1,W*H*D)
        s =  torch.bmm(f.permute(0,2,1),g)
        bta = self.softmax(s) 
        h = self.Wh(x).view(batch_size,-1,W*H*D)

        out = self.Wv(torch.bmm(h,bta.permute(0,2,1)).view(batch_size,self.CBar,W ,H, D))
        
        out = self.gamma*out + x
        
        return out
    


class bn3d(nn.Module):
  def __init__(self, output_size,  eps=1e-5, momentum=0.1,
                cross_replica=False, mybn=False):
    super(bn3d, self).__init__()
    self.output_size= output_size
    # Prepare gain and bias layers
    self.gain = P(torch.ones(output_size), requires_grad=True)
    self.bias = P(torch.zeros(output_size), requires_grad=True)
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Use cross-replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    
    if self.cross_replica:
      # not implemented, so error if requested
      assert 1==2
#      self.bn = SyncBN3d(output_size, eps=self.eps, momentum=self.momentum, affine=False)    
    elif mybn:
      # not implemented, so error if requested
      assert 1==2
#      self.bn = myBN(output_size, self.eps, self.momentum)
     # Register buffers if neither of the above
    else:     
      self.register_buffer('stored_mean', torch.zeros(output_size))
      self.register_buffer('stored_var',  torch.ones(output_size))
    
  def forward(self, x, y=None):
    if self.cross_replica or self.mybn:
      gain = self.gain.view(1,-1,1,1)
      bias = self.bias.view(1,-1,1,1)
      return self.bn(x, gain=gain, bias=bias)
    else:
      return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                          self.bias, self.training, self.momentum, self.eps)

class bigGAN_gen(nn.Module):
  def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128,
               G_kernel_size=3, G_attn='64', n_classes=1,
               num_G_SVs=1, num_G_SV_itrs=1,
               G_shared=True, shared_dim=0, hier=False,
               cross_replica=False, mybn=False,
               G_activation=nn.ReLU(inplace=False),
               G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
               BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
               G_init='ortho', skip_init=False, no_optim=False,
               G_param='SN', norm_style='bn',
               **kwargs):
    super(bigGAN_gen, self).__init__()
    # Channel width mulitplier
    self.ch = G_ch
    # Dimensionality of the latent space
    self.nz = dim_z
    # The initial spatial dimensions
    self.bottom_width = bottom_width
    # Resolution of the output
    self.resolution = resolution
    # Kernel size?
    self.kernel_size = G_kernel_size
    # Attention?
    self.attention = G_attn
    # number of classes, for use in categorical conditional generation
    self.n_classes = n_classes
    # Use shared embeddings?
    self.G_shared = G_shared
    # Dimensionality of the shared embedding? Unused if not using G_shared
    self.shared_dim = shared_dim if shared_dim > 0 else dim_z
    # Hierarchical latent space?
    self.hier = hier
    # Cross replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    # nonlinearity for residual blocks
    self.activation = G_activation
    # Initialization style
    self.init = G_init
    # Parameterization style
    self.G_param = G_param
    # Normalization style
    self.norm_style = norm_style
    # Epsilon for BatchNorm?
    self.BN_eps = BN_eps
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # fp16?
    self.fp16 = G_fp16
    # Architecture dict
    self.arch = biggan_G_arch3D(self.ch, self.attention)[resolution]
    print(self.arch)

    # If using hierarchical latents, adjust z
    if self.hier:
      # Number of places z slots into
      self.num_slots = len(self.arch['in_channels']) + 1
      self.z_chunk_size = (self.nz // self.num_slots)
      # Recalculate latent dimensionality for even splitting into chunks
      self.nz = self.z_chunk_size *  self.num_slots
    else:
      self.num_slots = 1
      self.z_chunk_size = 0

    # Which convs, batchnorms, and linear layers to use
    if self.G_param == 'SN':
      self.which_conv = functools.partial(SNConv3d,
                          kernel_size=3, padding=1,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(SNLinear,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
    else:
      self.which_conv = functools.partial(nn.Conv3d, kernel_size=3, padding=1)
      self.which_linear = nn.Linear
      
    # We use a non-spectral-normed embedding here regardless;
    # For some reason applying SN to G's embedding seems to randomly cripple G
    self.which_embedding = nn.Embedding
    bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                 else self.which_embedding)
    self.which_bn = functools.partial(ccbn3d,
                          which_linear=bn_linear,
                          cross_replica=self.cross_replica,
                          mybn=self.mybn,
                          input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                                      else self.n_classes),
                          norm_style=self.norm_style,
                          eps=self.BN_eps)


    # Prepare model
    # If not using shared embeddings, self.shared is just a passthrough
    self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared 
                    else identity())
    # First linear layer
    self.linear = self.which_linear(self.nz // self.num_slots,
                                    self.arch['in_channels'][0] * (self.bottom_width **3))

    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    # while the inner loop is over a given block
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      if index == 0:
        self.blocks += [[GBlock(in_channels=self.arch['in_channels'][index],
                               out_channels=self.arch['out_channels'][index],
                               which_conv=self.which_conv,
                               which_bn=self.which_bn,
                               activation=self.activation,
                               upsample=(functools.partial(F.interpolate, scale_factor=(1,2,2))
                                         if self.arch['upsample'][index] else None))]]
      else:
        self.blocks += [[GBlock(in_channels=self.arch['in_channels'][index],
                               out_channels=self.arch['out_channels'][index],
                               which_conv=self.which_conv,
                               which_bn=self.which_bn,
                               activation=self.activation,
                               upsample=(functools.partial(F.interpolate, scale_factor=2)
                                         if self.arch['upsample'][index] else None))]]

      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [Attention3d(self.arch['out_channels'][index], self.which_conv)]

    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    # output layer: batchnorm-relu-conv.
    # Consider using a non-spectral conv here
    self.output_layer = nn.Sequential(bn3d(self.arch['out_channels'][-1],
                                                cross_replica=self.cross_replica,
                                                mybn=self.mybn),
                                    self.activation,
                                    self.which_conv(self.arch['out_channels'][-1], 1))

    # Initialize weights. Optionally skip init for testing.
    if not skip_init:
      self.init_weights()

    # Set up optimizer
    # If this is an EMA copy, no need for an optim, so just return now
    if no_optim:
      return
    self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
#    if G_mixed_precision:
#      print('Using fp16 adam in G...')
#      import utils
#      self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
#                           betas=(self.B1, self.B2), weight_decay=0,
#                           eps=self.adam_eps)
#    else:
#      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
#                           betas=(self.B1, self.B2), weight_decay=0,
#                           eps=self.adam_eps)

    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d) 
          or isinstance(module, nn.Linear) 
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for G''s initialized parameters: %d' % self.param_count)

  # Note on this forward function: we pass in a y vector which has
  # already been passed through G.shared to enable easy class-wise
  # interpolation later. If we passed in the one-hot and then ran it through
  # G.shared in this forward function, it would be harder to handle.
  def forward(self, z, y=None):
    # if not proviced y, assume single class and build y ourselves
    if y == None:
        y = torch.zeros(z.shape[0],dtype=torch.int64).to(z.device)
      
    # If hierarchical, concatenate zs and ys
    if self.hier:
      zs = torch.split(z, self.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * len(self.blocks)
      
    # First linear layer
    h = self.linear(z)
    # Reshape
    h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width, self.bottom_width)
    
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      # Second inner loop in case block has multiple layers
      for block in blocklist:
        h = block(h, ys[index])
        
    # Apply batchnorm-relu-conv-tanh at output
    return torch.tanh(self.output_layer(h))

