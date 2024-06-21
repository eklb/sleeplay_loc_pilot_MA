# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:45:11 2020

@author: Moneta (--adapted by Kolbe)

Here we extract ROIs
ROIs: mOFC, lOFC, OFC, HP_all, HP_gm, EC, HP_all_EC, HP_gm_EC
Thresholds: 0.1,0.3,0.5 (relative to global signal)
smoothed: 4mm (for now, can do 8mm later)
Output: 
    1. ROI map per subject
    2. Signal loss per subject
    3. save data from ROI

background: defaults.mask.thresh = 0.8 will only estimate the model in voxels whose mean value is at least 80% of the global signal


Steps: 
    1. load the freesurfer in native space (can also do MNI later)
    2. take ROIs
    3. load the functional smoothed images and calculate the global mean signal for all data in block
    5. Get the mask of the brain that is at threshold
    6. See how much signal lose we have in the ROIs for each subject
    - it could be that we need to run signal clean for optimal? 
    
    labels i think : https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    OFC
    1012 2012    ctx-rh-lateralorbitofrontal
    1014 2014    ctx-rh-medialorbitofrontal
    HP
    17  Left-Hippocampus                        220 216 20  0
    53  Right-Hippocampus                       220 216 20  0
    1016    ctx-lh-parahippocampal              20  220 60  0
    2016    ctx-rh-parahippocampal              20  220 60  0
    EC
    1006    ctx-lh-entorhinal                   220 20  10  0
    2006    ctx-rh-entorhinal                   220 20  10  0
    MOTOR control
    1024    ctx-lh-precentral                   60  20  220 0
    2024    ctx-rh-precentral                   60  20  220 0
    
    1022    ctx-lh-postcentral                  220 20  20  0
    2022    ctx-lh-postcentral                  220 20  20  0
    
   primary auditory 
    1030    ctx-lh-superiortemporal             140 220 220 0
    2030    ctx-lh-superiortemporal             140 220 220 0


    ALSO - make brain mask as the intercept of all runs in session brain masks
"""

import os
#from os.path import join as opj
#import sys
import numpy as np
#import pandas as pd
from nilearn.image import load_img, new_img_like
#from nilearn.masking import apply_mask
#from nilearn.signal import clean
#import nibabel as nib
from pathlib import Path
import glob

# ========================================================================
# DEFINE ALL PATHS:
# ========================================================================

#run_comp=2 # 1 = cluster, 2 = local
path_root = os.getcwd()
if path_root[:12]=='/Users/kolbe':
    path_project = '/Users/kolbe/Documents/MPIB_NeuroCode/Sleeplay/data_analysis/MRI_loc_data'
    fold_suffix = 'fmri_prep'
else:
    path_project = '/home/mpib/kolbe/sleeplay_tardis/b_loc_v2/data'
    fold_suffix = 'derivatives'
    
#path_bids = os.path.join(path_project, 'bids')
path_fmriprep = os.path.join(path_project, fold_suffix)
path_output_files = os.path.join(path_project, 'masks')
path_output_info = os.path.join(path_project, 'masks')

#   # if running on the cluster, get relevant details from the bash runner file
# if run_comp==1:
#     #  get subject id:
#     #sub = 'sub-%s' % sys.argv[1]
#     #ses = 'ses-%s' % sys.argv[2]
#     sub = 'sub-02'
#     ses_all = ['ses-01', 'ses-02', 'ses-03', 'ses-04']
   
# # if not running on the cluster, define decoding settings here
# else:
#     # get subject id:
#     sub = 'sub-02'
#     # get session id:
#     ises = 0
#     ses = 'ses-01'

# ========================================================================
# DEFINE IMPORTANT VARIABLES:
# ========================================================================
sub_all = ['sub-01', 'sub-02', 'sub-03']
ses_all = ['ses-01', 'ses-02', 'ses-03', 'ses-04']
spaces = ['T1w'] #, 'MNI152NLin2009cAsym'

for isub, sub in enumerate(sub_all):
        
    for ises, ses in enumerate(ses_all):
    
        path_output_f = os.path.join(path_output_files, sub, ses, 'ROIs')
        path_output_i = os.path.join(path_output_info, sub, ses, 'Info')
        
        # create specified paths
        Path(path_output_f).mkdir(parents=True, exist_ok=True)
        Path(path_output_i).mkdir(parents=True, exist_ok=True)
        # outF = path_output_f
        # outFsl = path_output_i
    
        sub_data = list()
        #for sub in sublist:   
        print('starting ' + sub)
    
        # save nifti
        save_nifti = True
    
    
        do_rs = 0 # resting state
        # set number of runs/session
        RUNS = 12 # for object sessions
        if ses == 'ses-03' or ses == 'ses-04': # scene sessions with 10 runs
            RUNS = 10
    
        # if resting state
        #do_rs=0
    
        for space in spaces:
                print('starting space ' + space)
                
                '''
                ======================
                getting whole brain mask (across all runs)
                is normally used for univariate contrasts/search light? 
                ======================
                '''
                
                data_brain_mask = os.path.join(path_fmriprep, sub, ses, 'func', '*task-*' + space + '_desc-brain_mask.nii.gz')
                # sort all brain_masks in list: for each run per session per subject in specified space 
                data_brain_mask = sorted(glob.glob(data_brain_mask), key=lambda f: os.path.basename(f))
                
                # load whole brain mask for all runs 
                BM = load_img(data_brain_mask)
                # save as np.array 
                BMd = np.array(BM.dataobj).astype(int)
                BMs = np.sum(BMd, axis=3) # take the sum across run dimension
                if do_rs!=0:
                    # save brain mask
                    masked_image = new_img_like(BM, BMs>0)
                    if save_nifti: masked_image.to_filename(os.path.join(path_output_f,'BrainMask_Union_' + sub + '_' + ses + '_space-' + space + '.nii'))
                    
                    masked_image = new_img_like(BM, BMs==RUNS) # brain mask joint across all runs
                    if save_nifti: masked_image.to_filename(os.path.join(path_output_f,'BrainMask_intersect_' + sub + '_' + ses + '_space-' + space + '.nii'))
                
                
                for run in np.arange(1,RUNS+1):  
                    print('starting run ' + str(run))
                    if run<10:
                        crun='0'
                    else:
                        crun=''
                    #crun=''
                    # ROI masks       
                    
                    # ======================
                    # select ROIs
                    # ======================
                    
                    # load FreeSurfer segmentation volume (aligned to native T1)
                    data_FS = os.path.join(path_fmriprep, sub, ses, 'func', '*task-*_run-' + str(crun) + str(run) + '_space-' + space + '_desc-aparcaseg_dseg.nii.gz')
                    data_FS = sorted(glob.glob(data_FS), key=lambda f: os.path.basename(f))
                    print('data_FS:')
                    print(data_FS)
                    FS_seg_img = load_img(data_FS)
                    FS = np.array(FS_seg_img.dataobj).astype(int)
                    
                    # 1000 is left 2--- is right
                    #### visual cortex
                    VIS = (FS==1005) | (FS==2005) | (FS==1011) | (FS==2011) | (FS==1021) | (FS==2021) | (FS==1029) | (FS==2029) | (FS==1013) | (FS==2013) | (FS==1008) | (FS==2008) | (FS==1007) | (FS==2007) | (FS==1009) | (FS==2009) | (FS==1016) | (FS==2016) | (FS==1015) | (FS==2015)
                    mask_visual_labels = [
                    1005, 2005,  # cuneus
                    1011, 2011,  # lateral occipital
                    1021, 2021,  # pericalcarine
                    1029, 2029,  # superioparietal
                    1013, 2013,  # lingual
                    1008, 2008,  # inferioparietal
                    1007, 2007,  # fusiform
                    1009, 2009,  # inferiotemporal
                    1016, 2016,  # parahippocampal
                    1015, 2015,  # middle temporal
                    ]
                    
                    #### MTL + subregions
    
                    MTL =  (FS==17) | (FS==53) | (FS==1016) | (FS==2016) | (FS==1006) | (FS==2006)
                    
                    HPC =  (FS==17) | (FS==53) # 17 is left hippo 53 is right 
                    HPPHC =  (FS==17) | (FS==53) | (FS==1016) | (FS==2016) # HPC + paraHPC
                    PHPC = (FS==1016) | (FS==2016) # paraHPC
                    ERC = (FS==1006) | (FS==2006) #enthorhinal cortex
                    
                    #### posterior parietal cortex 
                    PPC = (FS==1029) | (FS==2029) | (FS==1008) | (FS==2008) #PPC = SPL + IPL
                    #SPL = (FS==1029) | (FS==2029)
                    IPL = (FS==1008) | (FS==2008) # inferior parietal lobe

                    PREC = (FS==1025) | (FS==2025) # precuneus (part of SPL)
                    
                    #### dorso-lateral PFC (use rostral + caudal middlefrontal gyrus as proxy)
                    DLPFC = (FS==1003) | (FS==2003) | (FS==1027) | (FS==2027)
                    
                    #### occipito-temporal regions: cuneus, lateral occipital sulcus, pericalcarine gyrus, superior parietal lobule, lingual gyrus, inferior parietal lobule, fusiform gyrus, inferior temporal gyrus, parahippocampal gyrus, and the middle temporal gyrus
                    OTC = (FS==1005) | (FS==2005) | (FS==1011) | (FS==2011) | (FS==1021) | (FS==2021) | (FS==1029) | (FS==2029) | (FS==1013) | (FS==2013) | (FS==1008) | (FS==2008) | (FS==1007) | (FS==2007) | (FS==1009) | (FS==2009) | (FS==1016) | (FS==2016) | (FS==1015) | (FS==2015)
                    
                    #### motor cortex as control region
                    MOT = (FS==1022) | (FS==2022)
                    
                    ### insula as second control
                    INS = (FS==1035) | (FS==2035)
    
                   
                    #### how many voxels per ROI
                    [print(np.sum(roi)) for roi in [VIS, MTL, HPC, HPPHC, PHPC, ERC, PPC, PREC, IPL, DLPFC, OTC, MOT, INS]] #VIS, MTL, HPC, HPPHC, PHPC, ERC, PPC, PREC, IPL, MOT
                    
    
                
                    # load func bold data and get global mean 
                    data_task = os.path.join(path_fmriprep, sub, ses, 'func',
                                             '*_task-*_run-' + str(crun) + str(run) + '_space-' + space + '_desc-preproc_bold.nii.gz')
                    data_task = sorted(glob.glob(data_task), key=lambda f: os.path.basename(f))
                    FS_task_img = load_img(data_task)
                    TASK = np.array(FS_task_img.dataobj)
                    
                    glob_mean = np.mean(TASK)
                    # calculate voxel mean
                    vox_mean = np.mean(TASK, axis=3)
                    #Voxmean=np.random.rand(FS.shape[0],FS.shape[1],FS.shape[2])
                    
                    #loop through thresholds
                    for thresh in [.0, .3, .5, .8]:
                        VOX_thresh = vox_mean>(thresh*glob_mean)
                        # save brain mask with voxels at thresh
                        masked_image = new_img_like(FS_seg_img,VOX_thresh)
                        if save_nifti:
                            outname=list()
                            outname=os.path.join(path_output_f, 'wholebrain_' + sub + '_' + ses + '_space-' + space + '_T-' + str(int(thresh*100)) + '_run-' + str(crun) + str(run) + '.nii')
                            if do_rs==1:
                                outname=list()
                                outname=os.path.join(path_output_f, 'wholebrain_' + sub + '_' + ses + '_task-rest_space-' + space + '_T-' + str(int(thresh*100)) + '_run-' + str(crun)  + str(run) + '.nii')
                            #masked_image.to_filename(os.path.join(outF,'wholebrain_'+sub+'_'+ses+'_space-'+space+'_T-'+str(int(T*100))+'_run-0'+str(run)+'.nii'))
                            masked_image.to_filename(outname)
                        # save number of voxels in whole brain
                        sub_data.append([sub, ses, run, space, 'WholeBrain', str(int(thresh*100)), np.sum(VOX_thresh), 1, thresh]) # 100%of anat as this is anat
    
                       #VIS, MTL, HPC, HPPHC, PHPC, ERC, PPC, PREC, IPL, DLPFC, OTC, MOT
                        for roi, roi_name in zip([INS], #VIS, MTL, HPC, HPPHC, PHPC, ERC, PPC, PREC, IPL, DLPFC, OTC, MOT
                                ['INS']): # 'VIS', 'MTL', 'HPC', 'HPPHC', 'PHPC', 'ERC', 'PPC', 'PREC', 'IPL', 'DLPFC', 'OTC','MOT'
    
                            # save ROI mask
                            masked_image = new_img_like(FS_seg_img,VOX_thresh& np.squeeze(roi))
                            if save_nifti: 
                                
                                masked_image.to_filename(os.path.join(path_output_f, roi_name + '_' + sub + '_' + ses + '_space-' + space + '_T-' + str(int(thresh*100)) + '_run-' + str(crun) + str(run) + '.nii'))
                            # save signal loss in ROI
                            sub_data.append([sub, ses, run, space, roi_name, str(int(thresh*100)), np.sum(VOX_thresh& np.squeeze(roi)), np.sum(VOX_thresh& np.squeeze(roi))/np.sum(roi),thresh]) # 100%of anat as this is anat
    
        np.savetxt(os.path.join(path_output_i,"ROIsDropout_add_INS_"+sub+'_'+ses+".csv"), np.vstack(sub_data), delimiter=',',fmt = '%s')     
