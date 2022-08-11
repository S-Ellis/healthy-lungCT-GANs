#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Sam Ellis
Copyright 2022 Sam Ellis, King's College London
"""
import numpy as np
import umap
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='', help="Path to UMAP_visualisation folder to use")
opt = parser.parse_args()

# load the latent vectors
w_background = np.load(f'{opt.outf}w_vectors_background.npy')
w_examples = np.load(f'{opt.outf}w_vectors_examples.npy')

branch_points_examples = np.load(f'{opt.outf}branch_stats_examples.npy')

# calculate the UMAP transformaation with the large number of background points
trans = umap.UMAP(n_neighbors=5, random_state=42).fit(w_background)

#%% map example points
w_examples_embedded = trans.transform(w_examples)

#%% plot colour coded to branch points
branch_stats = np.load(f'{opt.outf}branch_stats_examples.npy')

branch_stats_color = branch_stats
plt.scatter(w_examples_embedded[:,0],w_examples_embedded[:,1],c=branch_stats_color,cmap='gnuplot')
cbar = plt.colorbar()
plt.xlabel('UMAP dimension 1')
plt.ylabel('UMAP dimension 2')
plt.clim(0,120)
cbar.set_label('Number of branch points')
plt.show()

