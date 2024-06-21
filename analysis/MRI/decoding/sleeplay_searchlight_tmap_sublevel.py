#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:46:07 2023

@author: kolbe
"""

import sys
import os
import itertools
import glob
import copy
#matplotlib inline
import numpy as np
import nibabel as nb
from nilearn import image, masking, signal, plotting, surface, datasets
from nilearn.plotting import plot_img, plot_stat_map, show
from nilearn.decoding import SearchLight
from nilearn.image import resample_to_img, resample_img, new_img_like, smooth_img, threshold_img
from nilearn.maskers import NiftiMasker
from nilearn.glm import threshold_stats_img
from nilearn.datasets import load_mni152_template
from scipy.ndimage import binary_dilation
from scipy.signal import detrend
from scipy.stats import zscore
import matplotlib.pyplot as plt

import pandas as pd
import time
# from tqdm import tqdm
#import nipype.interfaces.fsl as fsl


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
save_dir = os.path.join(path_project, 'searchlight', 'plots')


    
# save_dir = os.path.join(path_project, 'searchlight')
# save_dir_files = os.path.join(save_dir, 'files', f"{sub}")

    
all_sub = ['sub-02'] 
sub=all_sub[0]
all_cond = ['enc', 'ret'] #, 'shuffled']  
all_data = ['observed', 'permuted']
all_ses = ['ses-01']#, 'ses-03']#, 'ses-02', 'ses-03', 'ses-04'] 
enc_types = ['obj', 'sc']          
obj_ses = ['ses-01']#, 'ses-02']
sc_ses = ['ses-03']#, 'ses-04']    
nrep_perm = 50
post_smooth = False
thresh = True
csmooth = 4 # smoothing kernel to smooth before group average
plot_only = False


# for map_idx in range(0,50):
#%%    
# behav_dir = os.path.join(base_dir, 'behav', 'decod_prep_data') # path to behavioural data 
sl_obj_enc, sl_obj_ret = list(), list()
sl_sc_enc, sl_sc_ret = list(), list()

# for sub in all_sub:
#     # if sub == 'sub-03':
#     #     all_ses = ['ses-02', 'ses-03', 'ses-04'] 
        
for ses in all_ses:
                     
    for cond in all_cond:
        slmap_dir = os.path.join(path_project, 'searchlight', 'files', f"{sub}", 
                                  f"{ses}", f"{cond}", 'observed')

        for i in os.listdir(slmap_dir):
            if i.endswith('MNI_smooth.nii'):
                slmap_path = os.path.join(slmap_dir, i)
        
        if cond == 'enc':
            if ses == 'ses-01': #or ses == 'ses-02':
                sl_obj_enc.append(slmap_path)

            elif ses == 'ses-03': #or ses == 'ses-04':
                sl_sc_enc.append(slmap_path)
        
        else:
            if ses == 'ses-01': #or ses == 'ses-02':
                sl_obj_ret.append(slmap_path)
                
            elif ses == 'ses-03': # or ses == 'ses-04':
                sl_sc_ret.append(slmap_path)

           
perm_obj_enc, perm_obj_ret = np.empty((1, nrep_perm), dtype=object), np.empty((1, nrep_perm), dtype=object)
perm_sc_enc, perm_sc_ret = np.empty((1, nrep_perm), dtype=object), np.empty((1, nrep_perm), dtype=object)
    
for enc_type in enc_types:
    # isub=0
    for sub in all_sub:
        isub=0
        # if sub == 'sub-03':
        #     obj_ses = ['ses-02']
        
        if enc_type == 'obj':
            nses = obj_ses
        else:
            nses = sc_ses
        
        for cond in all_cond:
            for ises, ses in enumerate(nses):
                permuted_map_dir = os.path.join(path_project, 'searchlight', 'files', f"{sub}", 
                                         f"{ses}", f"{cond}", 'permuted')
                print(len(os.listdir(permuted_map_dir)), sub, ses, cond)
                
                for map_idx, file_name in sorted(enumerate(os.listdir(permuted_map_dir))):
                    if enc_type == 'obj' and cond  == 'enc':
                        perm_obj_enc[isub,map_idx] = os.path.join(permuted_map_dir, file_name)
                    elif enc_type == 'obj' and cond == 'ret':
                        perm_obj_ret[isub,map_idx] = os.path.join(permuted_map_dir, file_name)
                    elif enc_type == 'sc' and cond == 'enc':
                        perm_sc_enc[isub,map_idx] = os.path.join(permuted_map_dir, file_name)
                    elif enc_type == 'sc' and cond == 'ret':
                        perm_sc_ret[isub,map_idx] = os.path.join(permuted_map_dir, file_name)
                    # break
        # isub += 2


#%% LOAD DATA FOR SINGLE SUBJECT SESSION, NO GROUP AVG
# sub_ses = 0 #sub-01,ses-01
# obs_map = image.load_img(sl_obj_enc[sub_ses])
# perm_maps = perm_obj_enc[sub_ses, :]
# perm_maps_ = [image.load_img(file) for file in perm_maps]
# obs_map_ = obs_map.get_fdata()
# perm_maps_ = [perm_map.get_fdata() for perm_map in perm_maps_]

# sub = 'sub-02'
# ses = 'ses-01'

# path_func_scan = os.path.join(data_dir, sub, ses, 'func', '*task-*MNI152NLin6Asym_desc-preproc_bold.nii.gz')
# path_func_scan = sorted(glob.glob(path_func_scan), key=lambda f: os.path.basename(f)) # = paths

# # load func data nifti files of all runs
# func_data = [image.load_img(i) for i in path_func_scan]

        
#%% LOAD NIFTI FILES, GETDATA, CALCULATE AVERAGE GROUP ACC MAPS

all_observed = [sl_obj_enc, sl_obj_ret]#, sl_sc_enc, sl_sc_ret]

obs_mean_keys = ['avg_obj_enc', 'avg_obj_ret']#, 'avg_sc_enc', 'avg_sc_ret']
obs_means = dict()

# Load all nifti files, convert, average into group acc map
###-> smooth before group average
for obs_map, key in zip(all_observed, obs_mean_keys):
    obs_map = image.load_img(obs_map)
    if post_smooth:
        obs_map = [smooth_img(smap, fwhm=csmooth) for smap in obs_map]
    # obs_map = np.stack([img.get_fdata() for img in obs_map], axis=-1)
    obs_means[key] = obs_map
    # obs_means[key] = obs_map.get_fdata()

for key, val in zip(obs_means.keys(), obs_means.values()):
    val = np.squeeze(val)
    obs_means[key] = val 

niimg_shape = image.load_img(all_observed[0])
breakpoint()


# iterate over all permutation arrays
all_permutes = [perm_obj_enc, perm_obj_ret, perm_sc_enc, perm_sc_ret]

# create empty lists to save average acc maps
perm_mean_keys = ['avg_perm_obj_enc', 'avg_perm_obj_ret', 'avg_perm_sc_enc', 'avg_perm_sc_ret']

perm_means = dict()

for key in perm_mean_keys:
    perm_means[key] = list()

# load permuted acc maps 
for key, perm_map in zip(perm_mean_keys, all_permutes):
    # for idc, paths in enumerate(perm_map):
    perm_map_ = [image.load_img(path) for path in perm_map] #if path is not None]
    if post_smooth:
        perm_map_ = [smooth_img(perm_map, fwhm=csmooth) for perm_map in perm_map_]
    # perm_map_ = np.stack([img.get_fdata() for img in perm_map_], axis=-1)
    # perm_means[key].append(np.mean(perm_map_, axis=-1))
    perm_means[key] = [img.get_fdata() for img in perm_map_]
    
# =============================================================================
# for file in paths
#     perm_list = [image.load_img(path) for idx, path in np.ndenumerate(perm_map, axis=0) if path is not None]
#     perm_list = np.stack([img.get_fdata() for img in perm_list], axis=-1)
# 
#             
#                 print(len(path))
#     tmp = [image.load_img(path) for idx, path in np.ndenumerate(perm_map) if path is not None]
# 
#     for idx,file in np.ndenumerate(perm_map):
#         if path is not None:
#             perm_map[idx] = image.load_img(file)
#             
# # save mean arrays for all group acc maps 
# for key, perm_map in zip(perm_mean_keys,all_permutes):
#     # mean = [[] for _ in range(perm_map.shape[1])]
#     for idx in range(perm_map.shape[1]):
#         tmp_array = perm_map[:,idx]
#         tmp_array = np.stack([img.get_fdata() for img in tmp_array], axis=-1)
#         perm_means[] = np.mean(tmp_array, axis=-1)
# =============================================================================
                 

# np.max(avg_obj_enc)
# np.min(np.ma.masked_equal(avg_obj_enc, 0))

#%% LOAD FUNC DATA FOR VISUALIZATION

##load functional data for each sub to plot stat_map
# mean_func_data = list()
# for sub in all_sub:
#     all_ses = ['ses-01', 'ses-02', 'ses-03', 'ses-04']
#     if sub == 'sub-03':
#         all_ses = ['ses-02', 'ses-03', 'ses-04'] 
#     for ses in all_ses:

#     # functional MRI files 
# path_func_scan = os.path.join(data_dir, all_sub[0], all_ses[0], 'func', '*task-*MNI152NLin6Asym_desc-preproc_bold.nii.gz')
# path_func_scan = sorted(glob.glob(path_func_scan), key=lambda f: os.path.basename(f)) # = paths

# # mean_func_data = list(itertools.chain(*mean_func_data))
# # load all images
# mean_func_data = [image.load_img(img) for img in path_func_scan]

# # compute the mean image
# mean_func_data = image.mean_img(mean_func_data)

#%% CALCULATE P-VALUE MAPS 

# create empty lists as placeholder
pmap_keys = ['pmap_obj_enc', 'pmap_obj_ret', 'pmap_sc_enc', 'pmap_sc_ret']

all_pmaps = dict()

for key in pmap_keys:
    all_pmaps[key] = list()

# array to save the pmap shapes
pmap_shape = np.empty(len(all_pmaps), dtype=object) 

# iterate over observed group maps to use as templates for pmap shapes
for idx, obs_key in enumerate(obs_mean_keys):
    pmap_shape[idx] = np.empty(obs_means[obs_key].shape, dtype=object)
    
# save shapes in predefined values
for idx, (key, shape) in enumerate(zip(pmap_keys, pmap_shape)):
    all_pmaps[key] = shape
breakpoint()
    
##### calculate p-values for group acc map
# compare each observed value (from obs group acc map) with all permuted maps at same voxel index
for map_idx, (obs_key, perm_key, pmap_key) in enumerate(zip(obs_mean_keys, perm_mean_keys, pmap_keys)):
    for vox_idx, val in np.ndenumerate(obs_means[obs_key]):
        p_count = 0
        for i in range(perm_means[perm_key][0].shape[3]):
            perm_map = perm_means[perm_key][0][:,:,:,i]
            if perm_map[vox_idx] > val:
                p_count += 1
        p_val = p_count/perm_means[perm_key][0].shape[3]
        all_pmaps[pmap_key][vox_idx] = p_val
    #convert to float for saving
    all_pmaps[pmap_key] = all_pmaps[pmap_key].astype(float)
    breakpoint()
    zero_mask = np.logical_not(obs_means[obs_key])
    p_ma = np.ma.array(all_pmaps[pmap_key], mask=zero_mask)
    p_val_img = new_img_like(niimg_shape, p_ma)
    #save pmap
    nb.save(p_val_img, os.path.join(save_dir, 'pmap_' + obs_key + '_smooth_sub02.nii'))



###### calculate p-values for single sub ses
# pmap = np.zeros((obs_map_.shape))

# for vox_idx, val in np.ndenumerate(obs_map_):
#     p_count = 0
#     for perm_map in perm_maps_:
#         if perm_map[vox_idx] > val:
#             p_count += 1

#     p_val = p_count/len(perm_maps_)
#     pmap[vox_idx] = p_val


#threshold pmap=tmap
if thresh:
    for idx, (pmap_key, obs_key) in enumerate(zip(pmap_keys, obs_mean_keys)):
        tmap = copy.deepcopy(all_pmaps[pmap_key])
        tmap = tmap.astype(float) 
        
        for vox_idx, val in np.ndenumerate(tmap):
            if val > 5/50:#2.5/50:
                tmap[vox_idx] = 0
                # tmap[vox_idx] = np.nan
                
        # nan_mask = np.isnan(tmap)
        zero_mask = np.logical_not(obs_means[obs_key])
        
        # joint_mask = nan_mask | zero_mask
        ### mask NaNs in tmap
        ### mask obs_map_ with logical not operator
        ### intersects both arrays?
        #---> plot?
        
        # F_score results // p_values?
        p_ma = np.ma.array(tmap, mask=zero_mask)#joint_mask
        p_val_img = new_img_like(niimg_shape, p_ma)
        #save tmap
        nb.save(p_val_img, os.path.join(save_dir, 'tmap_' + obs_key + '_smooth_sub02.nii'))

t1 = time.time()
total = (t1-t0)/60
print('Total run time: ' + str(total) + ' mins') 

    # plot_stat_map(
    #     p_val_img,
    #     mean_func_data,
    #     title="P_Map",
    #     display_mode="z",
    #     cut_coords=[-9],
    #     colorbar=False,)            

#%% PLOT PLOT PLOT

# load p_map nifti (if new console)

if plot_only:
    t_map=False
    sub=False
    if t_map:
        template = load_mni152_template(resolution=2)
    
        # load_pmap = list()
        
        # pmap_dir = os.path.join(save_dir, 'pmap_' + obs_key + '_post_smooth.nii')
        # for file in os.listdir(save_dir):
        #     if file.startswith('pmap') and file.endswith('post_smooth.nii'):
        #         load_pmap.append(file)
                
        # pmap_keys = ['pmap_obj_enc', 'pmap_obj_ret', 'pmap_sc_enc', 'pmap_sc_ret']
    
        # all_pmaps = dict()
        # for file in load_pmap:
        #     if file[9:-16] == 'obj_enc':
        #         all_pmaps['pmap_obj_enc'] = file
        #     elif file[9:-16] ==  'obj_ret':
        #         all_pmaps['pmap_obj_ret'] = file
        #     elif file[9:-16] == 'sc_enc':
        #         all_pmaps['pmap_sc_enc'] = file
        #     elif file[9:-16] == 'sc_ret':
        #         all_pmaps['pmap_sc_ret'] = file
        
        load_tmap = list()
        
        # tmap_dir = os.path.join(save_dir, 'tmap0_' + obs_key + 'smooth.nii')
        for file in os.listdir(save_dir):
            if file.startswith('tmap') and file.endswith('_sub02.nii'):
                load_tmap.append(file)
                
    
        tmap_keys = ['tmap_obj_enc', 'tmap_obj_ret', 'tmap_sc_enc', 'tmap_sc_ret']
    
        all_tmaps = dict()
        for file in load_tmap:
            if file[9:-17] == 'obj_enc':
                all_tmaps['tmap_obj_enc'] = file
            elif file[9:-17] ==  'obj_ret':
                all_tmaps['tmap_obj_ret'] = file
            elif file[9:-17] == 'sc_enc':
                all_tmaps['tmap_sc_enc'] = file
            elif file[9:-17] == 'sc_ret':
                all_tmaps['tmap_sc_ret'] = file
        
        for tmap_key in tmap_keys:
            all_tmaps[tmap_key] = image.load_img(os.path.join(save_dir, all_tmaps[tmap_key]))
        
        thresh_map=threshold_img(all_tmaps[tmap_key], threshold=0.001, cluster_threshold=200, copy=True)
        vmax=0.05
    elif sub:
        sl_obj_enc, sl_obj_ret = list(), list()
        sl_sc_enc, sl_sc_ret = list(), list()

        # for sub in all_sub:
        #     # if sub == 'sub-03':
        #     #     all_ses = ['ses-02', 'ses-03', 'ses-04'] 
                
        for ses in all_ses:
                             
            for cond in all_cond:
                slmap_dir = os.path.join(path_project, 'searchlight', 'files', f"{sub}", 
                                          f"{ses}", f"{cond}", 'observed')

                for i in os.listdir(slmap_dir):
                    if i.endswith('MNI_smooth.nii'):
                        slmap_path = os.path.join(slmap_dir, i)
                
                if cond == 'enc':
                    if ses == 'ses-01': #or ses == 'ses-02':
                        sl_obj_enc.append(slmap_path)

                    elif ses == 'ses-03': #or ses == 'ses-04':
                        sl_sc_enc.append(slmap_path)
                
                else:
                    if ses == 'ses-01': #or ses == 'ses-02':
                        sl_obj_ret.append(slmap_path)
                        
                    elif ses == 'ses-03': # or ses == 'ses-04':
                        sl_sc_ret.append(slmap_path)

        all_observed = [sl_obj_enc, sl_obj_ret]#, sl_sc_enc, sl_sc_ret]
    
        obs_mean_keys = ['avg_obj_enc', 'avg_obj_ret']#, 'avg_sc_enc', 'avg_sc_ret']
        obs_means = dict()
    
        # Load all nifti files, convert, average into group acc map
        ###-> smooth before group average
        for obs_map, key in zip(all_observed, obs_mean_keys):
            obs_map = image.load_img(obs_map)
            if post_smooth:
                obs_map = [smooth_img(smap, fwhm=csmooth) for smap in obs_map]
            # obs_map = np.stack([img.get_fdata() for img in obs_map], axis=-1)
            obs_means[key] = obs_map
            # obs_means[key] = obs_map.get_fdata()
    
        # for key, val in zip(obs_means.keys(), obs_means.values()):
        #     val = np.squeeze(val)
        #     obs_means[key] = val 

        #thresh=0.15 fro obj_enc
        thresh_map=threshold_img(obs_means['avg_obj_enc'], threshold=0.15, cluster_threshold=250, copy=True)
        vmax=None
        
    else:
        load_map=list()
        for file in os.listdir(save_dir):
            if file.startswith('group_acc') and file.endswith('_smooth.nii'):
                load_map.append(file)
        
        group_obj_enc = image.load_img(os.path.join(save_dir, load_map[1]))
        group_obj_ret = image.load_img(os.path.join(save_dir, load_map[0]))
        
        thresh_map=threshold_img(group_obj_enc, threshold=0.12, cluster_threshold=100, copy=True)
        thresh_map=threshold_img(group_obj_ret, threshold=0.1125, cluster_threshold=100, copy=True)
        threshold=0.1
        
    fig= plot_stat_map(
        thresh_map,
        # bg_img=template,
        display_mode="x",
        cut_coords=(-22, -24),
        # title="Threshold image with intensity value",
        # colorbar=False,
        cmap='cold_hot',
        threshold=threshold,
        vmax=None,
        black_bg=False
        )
    
    save_dir = '/Users/kolbe/Documents/MPIB_NeuroCode/Sleeplay/data_analysis/MRI_loc_data/searchlight/plots'
    
    fig.savefig(os.path.join(save_dir, 'thresh_acc_map_MNI_obj_enc'))
    
    # # Showing intensity threshold image
    # plot_stat_map(
    #     all_tmaps[tmap_key],
    #     bg_img=template,
    #     display_mode="x",
    #     cut_coords=5,
    #     title="Threshold image with intensity value",
    #     colorbar=False,
    #     )
    # texture = surface.vol_to_surf(all_tmaps[tmap_key], fsaverage.pial_right)
   
    # fig = plotting.plot_surf_stat_map(
    #     fsaverage.infl_right, texture, hemi='right',
    #     title='Surface right hemisphere', colorbar=True,
    #     threshold=.01, bg_map=curv_right_sign,
    #     )
    # # all_tmaps[tmap_key] = resample_to_img(all_tmaps[tmap_key], template)
    

    # thresh_img = threshold_stats_img(
    # os.path.join(save_dir, all_pmaps['pmap_obj_enc']), alpha=0.001, threshold=0.001, height_control=None)
    
    # thresh_img = threshold_img(
    # os.path.join(save_dir, all_pmaps['pmap_obj_enc']), threshold="95%", copy=True
    # )
    # tmap_keys = ['tmap_obj_enc', 'tmap_obj_ret', 'tmap_sc_enc', 'tmap_sc_ret']
    # all_tmaps = dict()
    # for key in tmap_keys:
    #     all_tmaps[key] = list()
        
    # for pmap_key, tmap_key in zip(pmap_keys, tmap_keys):
    #     all_pmaps[pmap_key] = image.load_img(os.path.join(save_dir, all_pmaps[pmap_key]))
    #     all_pmaps[pmap_key] = all_pmaps[pmap_key].get_fdata()
    #     zero_mask = np.logical_not(all_pmaps[pmap_key])
    #     p_ma = np.ma.array(all_pmaps[pmap_key], mask=zero_mask)

    #     pmap_thresh = (p_ma < 0.1)
    #     all_tmaps[tmap_key] = all_pmaps[pmap_key][pmap_thresh]
        
    # pmap_thresh = os.path.join(save_dir, all_pmaps['pmap_obj_enc']
    # thresh_img =    
    # # Showing intensity threshold image
    # plot_stat_map(
    #     thresh_img[0],
    #     display_mode="z",
    #     cut_coords=5,
    #     title="Threshold image with intensity value",
    #     colorbar=False,
    # )
  
    
    
    # p_map_img = image.load_img(os.path.join(save_dir_files, sl_nifti))

# # calculate mean functional image across all runs (as anatomical background image for plotting)
# mean_bold_img = image.mean_img(func_data)
# mean_bold_data = mean_bold_img.get_fdata()

# sl_all = ['enc', 'ret', 'enc+ret']

# for i in sl_all:
#     sl_nifti = "SearchLightMap_{sub}_{ses}_{i}.nii"

#     # load searchlight nifti (if new console)
#     searchlight_img = image.load_img(os.path.join(save_dir_files, sl_nifti))

#     # convert mean NIfTI into NumPy array
#     accuracy_map_data = searchlight_img.get_fdata()

#     # get min and max value from accuracy map (classification accuracy - chance level)/center voxel
#     sl_vmax = np.max(accuracy_map_data)
#     sl_vmin = np.min(np.ma.masked_equal(accuracy_map_data, 0)) # mask zeros within array to take minimum != 0

#     plot_img(
#         searchlight_img,
#         bg_img=mean_bold_img,
#         title="Searchlight",
#         display_mode="ortho",
#         cut_coords=None,
#         vmin=0.1,
#         vmax=0.18,
#         cmap="turbo",
#         threshold=0,
#         black_bg=True,
#     )


# Optionally, you can reshape the array if needed
# For example, if it's a 4D image and you want to reshape it into a 2D matrix
# new_shape = (img_data.shape[0] * img_data.shape[1], img_data.shape[2] * img_data.shape[3])
# img_data_reshaped = img_data.reshape(new_shape)

# Now you have the image data in a NumPy array
# You can perform various operations or analyses on this array


