#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:21:20 2024

@author: kolbe
"""

#%% LOAD AND DEFINE PATHS...
import os
import glob
import copy 
from nilearn import image, masking, signal, plotting
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from matplotlib import pyplot as plt
from matplotlib import colors 

#from matplotlib import ticker
import pickle
import itertools

local = True
# Define subject ID
all_sub = ['sub-02'] #'02'
nses = ['ses-01'] # , 'ses-02', 'ses-03', 'ses-04'

if local:
    base_dir = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'MPIB_NeuroCode', 
                                'Sleeplay', 'data_analysis')
    path_project = os.path.join(base_dir, 'MRI_loc_data')
    fmri_prep_suffix = os.path.join('fmri_prep')

else:
    base_dir = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'TARDIS', 'sleeplay_tardis')
    path_project = os.path.join(base_dir, 'b_loc_v2', 'data')
    fmri_prep_suffix = 'derivatives'

path_masks = os.path.join(path_project, 'masks')

mask_threshold=80

mask_labels =['VIS','MTL','HPC','PHPC','HPPHC','ERC','PPC','PREC','IPL','DLPFC','MOT'] #'PPC'

#%% Load all masks
for isub, sub in enumerate(all_sub):
    for ises, ses in enumerate(nses):
        
        print(f"Loading all masks for {sub} of {ses}")
    # load visual mask task files
        path_mask_VIS_scan1 = os.path.join(path_masks, 'MNI','VIS*T-' + str(mask_threshold) + '*nii')
        path_mask_VIS_scan1 = sorted(glob.glob(path_mask_VIS_scan1), key=lambda f: os.path.basename(f))
    
    # load medial temporal lobe mask task files
        path_mask_MTL_scan1 = os.path.join(path_masks, 'MNI','MTL*T-' + str(mask_threshold) + '*nii')
        path_mask_MTL_scan1 = sorted(glob.glob(path_mask_MTL_scan1), key=lambda f: os.path.basename(f))

    # load hippocampus mask task files
        path_mask_HPC_scan1 = os.path.join(path_masks, 'MNI','HPC*T-' + str(mask_threshold) + '*nii')
        path_mask_HPC_scan1 = sorted(glob.glob(path_mask_HPC_scan1), key=lambda f: os.path.basename(f))
    
    # load parahippocampus mask task files
        path_mask_PHPC_scan1 = os.path.join(path_masks, 'MNI','PHPC*T-' + str(mask_threshold) + '*nii')
        path_mask_PHPC_scan1 = sorted(glob.glob(path_mask_PHPC_scan1), key=lambda f: os.path.basename(f))
    
    # load hippo+parahippocampus mask task files:
        path_mask_HPPHC_scan1 = os.path.join(path_masks, 'MNI','HPPHC*T-' + str(mask_threshold) + '*nii')
        path_mask_HPPHC_scan1 = sorted(glob.glob(path_mask_HPPHC_scan1), key=lambda f: os.path.basename(f))

    # load entorhinal cortex mask task files
        path_mask_ERC_scan1 = os.path.join(path_masks, 'MNI','ERC*T-' + str(mask_threshold) + '*nii')
        path_mask_ERC_scan1 = sorted(glob.glob(path_mask_ERC_scan1), key=lambda f: os.path.basename(f))

    # load posterior parietal cortex mask task files
        path_mask_PPC_scan1 = os.path.join(path_masks, 'MNI','PPC*T-' + str(mask_threshold) + '*nii')
        path_mask_PPC_scan1 = sorted(glob.glob(path_mask_PPC_scan1), key=lambda f: os.path.basename(f))
    
    # load precuneus mask task files
        path_mask_PREC_scan1 = os.path.join(path_masks, 'MNI','PREC*T-' + str(mask_threshold) + '*nii')
        path_mask_PREC_scan1 = sorted(glob.glob(path_mask_PREC_scan1), key=lambda f: os.path.basename(f))

    # load inferior parietal cortex mask task files
        path_mask_IPL_scan1 = os.path.join(path_masks, 'MNI','IPL*T-' + str(mask_threshold) + '*nii')
        path_mask_IPL_scan1 = sorted(glob.glob(path_mask_IPL_scan1), key=lambda f: os.path.basename(f))

    # load PFC mask
        path_mask_DLPFC_scan1 = os.path.join(path_masks, 'MNI','DLPFC*T-' + str(mask_threshold) + '*nii')
        path_mask_DLPFC_scan1 = sorted(glob.glob(path_mask_DLPFC_scan1), key=lambda f: os.path.basename(f))
    
    # load motor cortex mask task files
        path_mask_MOT_scan1 = os.path.join(path_masks, 'MNI','MOT*T-' + str(mask_threshold) + '*nii')
        path_mask_MOT_scan1 = sorted(glob.glob(path_mask_MOT_scan1), key=lambda f: os.path.basename(f))
        
    # anatomical, functional mask
        path_anat_mask = os.path.join(path_project, fmri_prep_suffix, sub, 'anat', f"{sub}_desc-preproc_T1w.nii.gz") #/Users/kolbe/Documents/MPIB_NeuroCode/Sleeplay/data_analysis/MRI_loc_data/fmri_prep/sub-02/anat/sub-02_
        # path_anat_mask = sorted(glob.glob(path_anat_mask),key=lambda f: os.path.basename(f))
        
        path_func_mask = os.path.join(path_project, fmri_prep_suffix, sub, ses, 'func', '*task-*T1w_desc-brain_mask.nii.gz')
        path_func_mask = sorted(glob.glob(path_func_mask),key=lambda f: os.path.basename(f))


        print(f"loading scan 1 data of {sub} from {ses}")
        
        # load T1 mask:
        # anat = image.load_img(path_anat_scan1[0]) path needs updating (not actually used currently)
        
        ###loading overlay (intersect) masks for all runs per ROI 
        # load visual mask:
        mask_VIS_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_VIS_scan1)]
        mask_VIS_scan1 = masking.intersect_masks(mask_VIS_scan1, connected=False, threshold=0)
        # load mtl mask:
        mask_MTL_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_MTL_scan1)]
        mask_MTL_scan1 = masking.intersect_masks(mask_MTL_scan1, connected=False, threshold=0)
        # load hippocampus mask:
        mask_HPC_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_HPC_scan1)]
        mask_HPC_scan1 = masking.intersect_masks(mask_HPC_scan1, connected=False, threshold=0)
        # load parahippocampus mask:
        mask_PHPC_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_PHPC_scan1)]
        mask_PHPC_scan1 = masking.intersect_masks(mask_PHPC_scan1, connected=False, threshold=0)
        # load hippo+parahippocampus mask:
        mask_HPPHC_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_HPPHC_scan1)]
        mask_HPPHC_scan1 = masking.intersect_masks(mask_HPPHC_scan1, connected=False, threshold=0)
        # load entorhinal mask:
        mask_ERC_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_ERC_scan1)]
        mask_ERC_scan1 = masking.intersect_masks(mask_ERC_scan1, connected=False, threshold=0)
        # load ppc mask:
        mask_PPC_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_PPC_scan1)]
        mask_PPC_scan1 = masking.intersect_masks(mask_PPC_scan1, connected=False, threshold=0)
        # load prec mask:
        mask_PREC_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_PREC_scan1)]
        mask_PREC_scan1 = masking.intersect_masks(mask_PREC_scan1, connected=False, threshold=0)
        # load ipl mask:
        mask_IPL_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_IPL_scan1)]
        mask_IPL_scan1 = masking.intersect_masks(mask_IPL_scan1, connected=False, threshold=0)
        # load dlPFC mask:
        mask_DLPFC_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_DLPFC_scan1)]
        mask_DLPFC_scan1 = masking.intersect_masks(mask_DLPFC_scan1, connected=False, threshold=0)
        # load motor mask:
        mask_MOT_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_MOT_scan1)]
        mask_MOT_scan1 = masking.intersect_masks(mask_MOT_scan1, connected=False, threshold=0)
        
        # functional mask
        func_mask = [image.load_img(i) for i in path_func_mask]
        func_mask = masking.intersect_masks(func_mask, connected=False, threshold=0)

        # Load anat brain mask for each run for plotting
        anat_mask = image.load_img(path_anat_mask)
        # anat_mask = masking.intersect_masks(anat_mask, connected=False, threshold=0)
        
        all_masks = {'VIS': mask_VIS_scan1, #'lpfc': mask_lpfc_scan1, 
                       'MTL': mask_MTL_scan1,
                       'HPC': mask_HPC_scan1, 
                       'PHPC': mask_PHPC_scan1,
                       'HPPHC': mask_HPPHC_scan1,
                       'ERC': mask_ERC_scan1,
                       'PPC': mask_PPC_scan1,
                       'PREC': mask_PREC_scan1,
                       'IPL': mask_IPL_scan1,
                       'DLPFC': mask_DLPFC_scan1,
                       'MOT': mask_MOT_scan1
                       }

#%% Plot 
        for i in mask_labels:
            save_name =  f"{i}_ROI_plot.png"            
            plotting.plot_roi(all_masks[i], black_bg=False, draw_cross=False, title = i) #anat_mask,
            plt.savefig(os.path.join(path_masks, 'ROI_plots', save_name))
        # plotting.plot_stat_map(all_masks['VIS'], anat_mask)


        
    
