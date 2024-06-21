#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:38:19 2023

@author: kolbe
"""

import glob
import os
import numpy as np
import copy
from nilearn import image, masking, signal, plotting

# =============================================================================
### DEFINE ALL PATHS AND VARIABLES
# =============================================================================

base_dir = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'MPIB_NeuroCode', 
                            'Sleeplay', 'data_analysis')
path_project = os.path.join(base_dir, 'MRI_loc_data')
fmri_prep_suffix = os.path.join('fmri_prep')
path_behav = os.path.join(base_dir, 'behav', 'decod_prep_data')
path_masks = os.path.join(path_project, 'masks')
path_fmriprep = os.path.join(path_project, fmri_prep_suffix)
path_output_files = os.path.join(path_project, 'decoding', 'files')


sub = 'sub-02'
# all_sub = ['sub-01', 'sub-02', 'sub-03']
nses = ['ses-01', 'ses-02', 'ses-03', 'ses-04'] # ses-02 excluded for sub-02 (first run too short?)
# ses-02 ses-03 excluded since run = 7 too short?

mask_labels =['VIS','MTL', 'HPC','ERH']
TR=1.25
mask_threshold=80
whole_brain_smooth=1

ises = 0
ses = nses[ises]

enc_type = 'obj' # required to load behaviour
if ses == 'ses-03' or ses == 'ses-04': # scene sessions
    enc_type = 'sc'


# =============================================================================
### DEFINE PATHS TO LOAD DATA (e.g. MASKS) 
# =============================================================================

# load the visual mask task files
path_mask_vis_task_scan1 = os.path.join(path_masks,sub,ses,'ROIs','VIS*T-' + str(mask_threshold) + '*nii')
path_mask_vis_task_scan1 = sorted(glob.glob(path_mask_vis_task_scan1), key=lambda f: os.path.basename(f))

# load hippocampus mask task files
path_mask_hpc_task_scan1 = os.path.join(path_masks,sub,ses,'ROIs','HPC*T-' + str(mask_threshold) + '*nii')
path_mask_hpc_task_scan1 = sorted(glob.glob(path_mask_hpc_task_scan1), key=lambda f: os.path.basename(f))

# load ERH mask (entorhinal cortex)
path_mask_erh_task_scan1 = os.path.join(path_masks,sub,ses,'ROIs','ERH*T-' + str(mask_threshold) + '*nii')
path_mask_erh_task_scan1 = sorted(glob.glob(path_mask_erh_task_scan1), key=lambda f: os.path.basename(f))
    
# load PFC mask (prefrontal cortex)
path_mask_lpfc_task_scan1 = os.path.join(path_masks,sub,ses,'ROIs','LPFC*T-' + str(mask_threshold) + '*nii')
path_mask_lpfc_task_scan1 = sorted(glob.glob(path_mask_lpfc_task_scan1), key=lambda f: os.path.basename(f))
#print(path_mask_pfc_task_scan1)
 
# load MTL mask task files (medial temporal lobe)
path_mask_mtl_task_scan1 = os.path.join(path_masks,sub,ses,'ROIs','MTL*T-' + str(mask_threshold) + '*nii')
path_mask_mtl_task_scan1 = sorted(glob.glob(path_mask_mtl_task_scan1), key=lambda f: os.path.basename(f))

# load whole brain mask files
# path_mask_whole_brain_scan1 = os.path.join(path_fmriprep, sub, 'ses-01', 'func','*ses-01*task-gems*T1w*brain_mask.nii.gz')
path_mask_whole_brain_scan1=os.path.join(path_masks,sub,ses,'ROIs','wholebrain*T-' + str(mask_threshold) + '*nii')
path_mask_whole_brain_scan1 = sorted(glob.glob(path_mask_whole_brain_scan1), key=lambda f: os.path.basename(f))

# load the functional mri task files
path_func_task_scan1 = os.path.join(path_fmriprep,sub,ses,'func', '*task-*T1w_desc-preproc_bold.nii.gz')
path_func_task_scan1 = sorted(glob.glob(path_func_task_scan1), key=lambda f: os.path.basename(f))

# load the anatomical mri file:
path_anat_scan1 = os.path.join(path_fmriprep, sub,'anat','%s_desc-preproc_T1w.nii.gz' % sub)
# path_anat_scan1 = sorted(glob.glob(path_anat_scan1), key=lambda f: os.path.basename(f))


print('Loading all data...')

# load T1 mask:
anat = image.load_img(path_anat_scan1) 

###loading overlay (intersect) masks for all runs per ROI 
# load visual mask:
mask_vis_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_vis_task_scan1)]
mask_vis_scan1 = masking.intersect_masks(mask_vis_scan1, connected=False, threshold=0)
# load hippocampus mask:
mask_hpc_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_hpc_task_scan1)]
mask_hpc_scan1 = masking.intersect_masks(mask_hpc_scan1, connected=False, threshold=0)
# load entorhinal mask:
mask_erh_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_erh_task_scan1)]
mask_erh_scan1 = masking.intersect_masks(mask_erh_scan1, connected=False, threshold=0)
# # load PFC mask:
# mask_lpfc_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_lpfc_task_scan1)]
# mask_lpfc_scan1 = masking.intersect_masks(mask_lpfc_scan1, connected=False, threshold=0)
# load mtl mask:
mask_mtl_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_mtl_task_scan1)]
mask_mtl_scan1 = masking.intersect_masks(mask_mtl_scan1, connected=False, threshold=0)
# load wholebrain mask:
mask_wholebrain_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_whole_brain_scan1)]
mask_wholebrain_scan1 = masking.intersect_masks(mask_wholebrain_scan1, connected=False)

# define mask dictionary
masks_scan1 = {'VIS': mask_vis_scan1, #'lpfc': mask_lpfc_scan1, 
               'HPC': mask_hpc_scan1, 'MTL': mask_mtl_scan1,
               'wholebrain': mask_wholebrain_scan1, 'ERH': mask_erh_scan1}

# load functional data
data_task_scan1 = [image.load_img(i) for i in path_func_task_scan1]

# =============================================================================
### APPLY MASKS AND START SMOOTHING (WHOLE BRAIN VS. MASKED ROI)
# =============================================================================

# set kernel size (width of curve) as Full Width at Half Maximum (FWHM) for spatial smoothing
csmooth = 4

# create dicts for all mask timeseries
data_masked_WBsmooth = dict()
data_masked_ROIsmooth = dict()


# begin looping over masks (whole brain smoothing vs. ROI smoothing)
for imask, cmask in enumerate(mask_labels):
    print('Running ' + cmask + ' mask')
   
    # apply whole brain smoothing, then mask data based on a specific anatomical region
    if whole_brain_smooth==1:
        print('Running wholebrain smoothing and applying anatomical ' + cmask + ' mask')
        # apply mask for each ROI including wholebrain smoothing: get 2d timeseries data (TR number x voxel number)
        data_masked_scan1=[]
        data_masked_scan1 = [masking.apply_mask(i, masks_scan1[cmask], smoothing_fwhm=csmooth) for i in data_task_scan1]
        # # bring masked 2d timeseries data back into 4D space of mask for later comparison
        # unmasked_data_scan1 = []
        # unmasked_data_scan1 = [masking.unmask(i, masks_scan1[cmask]) for i in data_masked_scan1]
        # save list of all data per run in dict per mask
        data_masked_WBsmooth[cmask] = data_masked_scan1

    # mask data first and smooth only masked voxels afterwards
    #else:
        print('Applying anatomical ' + cmask + ' mask and smooth ROI voxels afterwards')
        # apply masks for each ROI without whole brain smoothing: get 2d timeseries data (TR number x voxel number)
        data_masked_scan1_nosmooth = []
        data_masked_scan1_nosmooth = [masking.apply_mask(i, masks_scan1[cmask]) for i in data_task_scan1]
        # bring masked 2d timeseries data back into 4D space of mask
        unmasked_data_scan1_nosmooth = []
        unmasked_data_scan1_nosmooth = [masking.unmask(i, masks_scan1[cmask]) for i in data_masked_scan1_nosmooth]
        # apply smoothing of ROI mask voxels 
        ROI_smoothed_data = []
        ROI_smoothed_data = [image.smooth_img(i, fwhm=csmooth) for i in unmasked_data_scan1_nosmooth]
        # to get timeseries data again
        data_masked_scan1_smoothROI = []
        data_masked_scan1_smoothROI = [masking.apply_mask(i, masks_scan1[cmask]) for i in ROI_smoothed_data]
        # save list of all data per run in dict per mask
        data_masked_ROIsmooth[cmask] = data_masked_scan1_smoothROI

# =============================================================================
### PLOT MASKS TO CHECK
# =============================================================================
# for key in masks_scan1:

#     plotting.plot_roi(masks_scan1[key], bg_img=anat) 


# =============================================================================
### (CENTRAL) VOXEL COMPARISON BETWEEN ROIs IN BOTH SMOOTHING CONDITIONS
# =============================================================================
# time series data: image (TR) number x voxel number 
TRmean_WBsmooth = dict()
TRmean_ROIsmooth = dict()
centr_vox_diff = dict()

for key in mask_labels:
    #calculate mean across TRs for first run only
    TRmean_WBsmooth[key]= mean_WBsmooth = data_masked_WBsmooth[key][0].mean(axis=0)
    TRmean_ROIsmooth[key] = mean_ROIsmooth = data_masked_ROIsmooth[key][0].mean(axis=0)
    # take 100 central voxel for each ROI
    centr_vox_WBsmooth = mean_WBsmooth[(mean_WBsmooth.shape[0]//2)-50:(mean_WBsmooth.shape[0]//2)+50]
    centr_vox_ROIsmooth = mean_ROIsmooth[(mean_ROIsmooth.shape[0]//2)-50:(mean_ROIsmooth.shape[0]//2)+50]
    #calculate signal difference for selected voxels in both smoothing conditions (WB-ROI)
    centr_vox_diff[key] = np.subtract(centr_vox_WBsmooth, centr_vox_ROIsmooth)
    # centr_vox_diff[key] = np.absolute(centr_vox_WBsmooth - centr_vox_ROIsmooth)


    
    

