#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:26:13 2024

@author: kolbe
"""

import sys
import os
import glob
import copy
import pickle
#matplotlib inline
import numpy as np
import pandas as pd
import time

# start timer
t0 = time.time()

# =============================================================================
# ### DEFINE IMPORTANT PATHS:
# =============================================================================
mounted = True

path_root = os.getcwd()
if path_root[:12]=='/Users/kolbe':
    run_clust = False
    base_dir = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'MPIB_NeuroCode', 
                                'Sleeplay', 'data_analysis')
    path_project = os.path.join(base_dir, 'MRI_loc_data')
    
    fmri_prep_suffix = os.path.join('fmri_prep')

    if mounted:
        base_dir = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'TARDIS',
                                'sleeplay_tardis')
        path_project = os.path.join(base_dir, 'b_loc_v2', 'data')
        
        fmri_prep_suffix = 'derivatives'
    
    
else:
    run_clust = True
    base_dir = os.path.join(os.sep, 'home', 'mpib', 'kolbe', 'sleeplay_tardis')
    path_project = os.path.join(base_dir, 'b_loc_v2', 'data')
    fmri_prep_suffix = 'derivatives'

data_dir = os.path.join(path_project, fmri_prep_suffix)
path_decod_files = os.path.join(path_project, 'decoding', 'files')

all_sub = ['sub-01', 'sub-02', 'sub-03']
all_ses = ['ses-01', 'ses-02', 'ses-03', 'ses-04'] 
all_data = ['observed', 'permuted'] 
all_cond = ['enc', 'ret', 'enc2ret'] 
incl_masks = ['VIS', 'MTL', 'HPC', 'ERC', 'PPC', 'PREC', 'IPL', 'MOT']

### LOAD ALL FILES: OBSERVED AND PERMUTED PER SUB
sub_list_obs = list()
sub_list_perm = list()
for isub, sub in enumerate(all_sub):
    for dat in all_data:
        file_dir = os.path.join(path_decod_files, f"{sub}", f"{dat}")
        if dat == 'observed':
            file_path = os.path.join(file_dir, 'ROI_decod_acc_loc.pkl')
            with open(file_path, "rb") as file:
                decod_dat = pickle.load(file)
            sub_list_obs.append(decod_dat)
        else:
            sub_list_perm.append(list())
            for i in os.listdir(file_dir):
                file_path = os.path.join(file_dir, i)
                with open(file_path, "rb") as file:
                    decod_dat = pickle.load(file)
                sub_list_perm[isub].append(decod_dat)

obs_masks = sub_list_obs[0]['acc_1dim_label']

# extract indices of masks from perm data for analyses
mask_idx = list()
for idx, mask in enumerate(obs_masks):
    for imask, mask_label in enumerate(incl_masks):
        if mask == mask_label:
            mask_idx.append(idx)

# create lists for all data_points to append p-values
p_sub = list()
for isub, sub_obs in enumerate(sub_list_obs):
    obs_dat = sub_obs['acc'][mask_idx, :, :3] #include only data from specified mask indices and from cond 1-3
    p_list = list()
    for perm_dat in sub_list_perm[isub]:
        p_val = perm_dat['acc'] > obs_dat
        p_list.append(p_val)
    
    p_list = np.stack(p_list)
    p_sum = np.sum(p_list, axis=0)/100
    p_sub.append(p_sum)

# p_sub = np.stack(p_sub)

thres_p_vals = list()
for i in p_sub:
    mask_thres = i > 0.1
    thres_p = np.ma.masked_array(i, mask=mask_thres)
    thres_p_vals.append(thres_p)

p_dict = dict()
for idx, (sub, thres_p, sub_id) in enumerate(zip(p_sub, thres_p_vals, all_sub)):
    if isub == 2:
        all_ses = all_ses = ['ses-02', 'ses-03', 'ses-04'] 
    # file_dir = os.path.join(path_decod_files, f"{sub}", 'permuted')
    file_dir = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'MPIB_NeuroCode', 
                                'Sleeplay', 'data_analysis', 'MRI_loc_data', 'decoding', 'plots')
    file_name = f"ROI_p_vals_{sub_id}.pkl"
    
    
    p_dict['p_val'] = sub
    p_dict['thres_p'] = thres_p
    p_dict['p_val_1dim_label'] = incl_masks
    p_dict['p_val_2dim_label'] = all_ses
    p_dict['p_val_3dim_label'] = all_cond
    
    
    # save main dict to pickle file
    with open(os.path.join(file_dir, file_name), "wb") as file:
        pickle.dump(p_dict, file, pickle.HIGHEST_PROTOCOL)
    
    
# thres_p_vals = np.stack(thres_p_vals)
     
# for isub, sub_obs in enumerate(sub_list_obs):
#     obs_dat = sub_obs['acc'][mask_idx, :, :3] #include only data from specified mask indices and from cond 1-3
#     for ises in range(len(all_ses)):
        
#         for mask_perm, mask_obs in enumerate(mask_idx):
#             for icond in len(all_cond):
#                 obs_dat = sub_obs['acc'][imask, ises, :3] #include only data from specified mask indices and from cond 1-3
#                 p_val = 0
#                 for iperm, perm_acc in enumerate(sub_list_perm[isub]):
#                     if perm_acc['acc'][mask_perm, ises, icond] > sub_obs['acc'][mask_obs, ises, icond]:
#                         p_val += 1







            
        
