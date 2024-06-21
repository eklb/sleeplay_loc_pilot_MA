#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:46:07 2023

@author: kolbe

"""
#%% PACKAGES AND FUNCTIONS
# ========================================================================
### IMPORT PACKAGES:
# ========================================================================

import glob
import os
import pandas as pd
import copy
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from nilearn import image, masking, signal#, plotting
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
#from imblearn.over_sampling import SMOTE
import time

# =============================================================================
### DEFINE ALL FUNCTIONS
# =============================================================================

def find_closest_TR(target, start=0, end=1000, tr=1.25):
    # Generate number stream
    numbers = []
    while start <= end:
        numbers.append(start)
        start += tr
    # find closest TR index
    closest_index = None
    closest_difference = float("inf")
    for i, num in enumerate(numbers):
        difference = abs(num - target)
        if difference < closest_difference:
            closest_index = i
            closest_difference = difference
    return closest_index

def pickNprep_TRs(func_data, time_idx, runs, run_vector, HRF):    
    # get relevant TRs from detrended data
    func_data_TRs = list()

    # find closest TR of each timestamp of cue presentation
    TR_idx = np.empty(len(time_idx))
    for i, t in enumerate(time_idx):
        TR_idx[i] = find_closest_TR(t, 0, 1000, TR)
    
    # add time shift of 4 TRs to tr_idx (for HRF response)
    #global TR_idx
    TR_idx = TR_idx + HRF
    
    # select data according to TRs of each run
    for irun in range(0,len(runs)):
        data_tmp = func_data[irun]
        idx_tmp = TR_idx[run_vector == irun+1].astype(int)
# =============================================================================
#         # create list with indices which are out of bounds if runs are too short
#         del_idc = [i for i,e in enumerate(idx_tmp) if e>len(data_tmp)]
#         for i,e in enumerate(idx_tmp):
#             if e>len(data_tmp):
#                 e = np.nan
#         if not del_idc == list():
#             # delete elements from index array
#             idx_tmp = np.delete(idx_tmp, del_idc)
#             print(f"Run {irun+1} is too short. Indices out of bounds will be deleted...")
# =============================================================================
        # if run too short, set idx_tmp to zero's for trials out of bound
        idx_tmp = np.where(idx_tmp>=len(data_tmp), 0, idx_tmp)
        trl_count = 0
        for i,e in enumerate(idx_tmp):
            if e==0:
                trl_count += 1
                tmp = np.empty(data_tmp.shape[1])
                tmp[:] = np.nan
                func_data_TRs.append(tmp)
            else:
                func_data_TRs.append(data_tmp[e])
            #func_data_TRs.append(data_tmp[idx_tmp])
        if trl_count != 0:
            print(f"Run {irun+1} of {ses}: A total of {trl_count} Trials are out of bound.")

    # combine (concatenate) the detrended & selected data of all runs:
    func_data_TRs = np.vstack(func_data_TRs)
    return func_data_TRs

def del_NaN_trls(nan_data, cat_labels, run_vector):
    '''
    Delete all trials of functional data, class labels and run vector from 
    shorter scan runs which are out of bounds. 
    '''
    # check for NaNs in functional data and delete trials (and corresponding labels/trials of run vector)
    if nan_data.ndim == 1:
        nan_idc = np.argwhere(np.isnan(nan_data))
    else:
        nan_idc = np.argwhere(np.isnan(nan_data[:, 0]))
    nan_idc = nan_idc.flatten('F')
    print(f"Deleting {len(nan_idc)} nan trials")
    nan_data_tmp = np.delete(nan_data, nan_idc, axis=0)
    label_tmp = np.delete(cat_labels, nan_idc)
    run_tmp = run_vector.drop(nan_idc)
    return nan_data_tmp, label_tmp, run_tmp

def Classification_TisT(func_data, cat_labels, runs, run_vector, clf):
    '''
    Leave-One-Out Cross-Validation when Classifier is trained and tested 
    on same cue types of functional data.
    '''
    #print('Training and testing classifier.')
    # loop over training runs
    run_list = np.arange(1,len(runs)+1)
    
    # initialise array for classification accuracy for each run 
    acc_tmp = np.empty(len(run_list))

    for irun in run_list: 
        ### Organise training and test data
        # leave-one-out cross-validation
        
        # define the run indices for the test set:
        test_runs = [irun]
        # define all remaining runs as train runs
        train_runs = runs[~pd.Series(runs).isin(test_runs)]
        
        # get training data
        train_data = func_data[run_vector.isin(train_runs)]
        
            
        #exlude all nans from train_data (AND test_data) + delete corresponding labels
        
        train_labels = cat_labels[run_vector.isin(train_runs)]
        
        # get testing data
        test_data = func_data[run_vector.isin(test_runs)]
        test_labels = cat_labels[run_vector.isin(test_runs)]
        
        # fit the model using the training data (e.g. train the model)
        clf.fit(train_data, train_labels)
        
        # predict testing classes & probabilities
        pred_class = clf.predict(test_data)
        pred_proba = clf.predict_proba(test_data)
        
        # calculate percent correct for predicted class labels of all runs
        acc_tmp[irun-1] = np.sum(test_labels == pred_class)/len(pred_class)
        
    return acc_tmp
        
            
def Classification_TisNotT(func_data_train, func_data_test, cat_labels_train, 
                           cat_labels_test, runs, run_vector_train, run_vector_test, clf):
    '''
    Leave-One-Out Cross-Validation when Classifier is trained and tested 
    on different cue types of functional data.
    '''
    
    #print('Training and testing classifier.')

    # loop over training runs
    run_list = np.arange(1,len(runs)+1)
    
    # initialise array for classification accuracy for each run 
    acc_tmp = np.empty(len(run_list))
    
    train_runs = runs
    # define training data (stays the same)
    train_data = func_data_train[run_vector_train.isin(train_runs)]
    train_labels = cat_labels_train[run_vector_train.isin(train_runs)]
    
    for irun in run_list: 
        ### Organise training and test data
        # leave-one-out cross-validation
        
        # define the run indices for the test set:
        test_runs = [irun]
        # # define all remaining runs as train runs
        # train_runs = runs[~pd.Series(runs).isin(test_runs)]
        
        # # get training data
        # train_data = func_data[run_vector.isin(train_runs)]
        # train_labels = cat_labels[run_vector.isin(train_runs)]
        
        # get testing data
        test_data = func_data_test[run_vector_test.isin(test_runs)]
        test_labels = cat_labels_test[run_vector_test.isin(test_runs)]
        
        # train classifier: fit model using the training data
        clf.fit(train_data, train_labels)
        
        # predict testing classes & probabilities
        pred_class = clf.predict(test_data)
        pred_proba = clf.predict_proba(test_data)

        acc_tmp[irun-1] = np.sum(test_labels == pred_class)/len(pred_class)
        
    return acc_tmp

#%% PATHS AND PARAMETERS
# start timer
t0 = time.time()

# ========================================================================
### DEFINE IMPORTANT PATHS:
# ========================================================================

path_root = os.getcwd()
if path_root[:12]=='/Users/kolbe':
    run_clust = False
    base_dir = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'MPIB_NeuroCode', 
                                'Sleeplay', 'data_analysis')
    path_project = os.path.join(base_dir, 'MRI_loc_data')
    fmri_prep_suffix = os.path.join('fmri_prep')
else:
    run_clust = True
    base_dir = os.path.join(os.sep, 'home', 'mpib', 'kolbe', 'sleeplay_tardis')
    path_project = os.path.join(base_dir, 'b_loc_v2', 'data')
    
    fmri_prep_suffix = 'derivatives'
    
path_behav = os.path.join(base_dir, 'behav', 'decod_prep_data') # path to behavioural data 
path_masks = os.path.join(path_project, 'masks')
#path_masks = '/Users/kolbe/Documents/TARDIS/sleeplay_tardis/b_loc_v2/data/masks'
path_fmriprep = os.path.join(path_project, fmri_prep_suffix)
path_output_files = os.path.join(path_project, 'decoding', 'files')
path_output_plots = os.path.join(path_project, 'decoding', 'plots')

# ========================================================================
### SET PARAMETERS:
# ========================================================================

#all_sub = ['sub-01', 'sub-02'] #'sub-03'

 # if running on the cluster, get sub ID from bash runner file
if run_clust:
    # get subject id:
    sub = 'sub-%s' % sys.argv[1]
    n_rep = str(sys.argv[2])

# if not running on the cluster, define sub ID here
else:
    sub = 'sub-02'
    
nses = ['ses-01', 'ses-02', 'ses-03', 'ses-04'] # ses01= objects_loc1, ses03 = scenes_loc1 
# --> ses-02 excluded for sub-02 (first run too short?)
# ses-02 ses-03 excluded since run = 7 too short?
if sub == 'sub-03':
    nses = ['ses-02', 'ses-03', 'ses-04'] 

# define trial type conditions
cue_type = ['enc', 'ret', 'enc2ret']#, 'enc&ret'] # where classifiers are trained/tested on 
mask_labels =['VIS','MTL','HPC','ERC','PPC','PREC','IPL','MOT'] #  #'PHPC','HPPHC', 'INS', 'DLPFC'
TR=1.25



### OUTPUT VARIABLES
# create empty dataframes to store data
#data_out = pd.DataFrame()
data_out = dict()

decod_acc = np.empty([len(mask_labels), len(nses), len(cue_type)])
decod_acc_all = copy.deepcopy(decod_acc)
decod_acc_all = [[e] for i,e in np.ndenumerate(decod_acc)]
decod_acc[:] = np.nan

save_name = f"ROI_decod_acc_loc_permuted{n_rep}.pkl" #'ROI_decod_acc_loc.pkl' # saved in sub specific folder

smooth = True # if False, no smoothing applied
whole_brain_smooth = True #--> True = wholebrain smoothing (also True for no smoothing), False = within mask smoothing
lbl_shuffle = True

# use masks thresholded based on global signal
use_thresholded_masks=1
# set threshold for the above 
# this will only include voxels with values x% of, or higher than the global signal mean
mask_threshold=80
       
#%% BEHAVIOURAL DATA, PATHS
# =============================================================================
# ises = 0
# ses = nses[ises]
# =============================================================================
#for isub, sub in enumerate(all_sub):
for ises, ses in enumerate(nses):
    enc_type = 'obj' # required to load behaviour
    if ses == 'ses-03' or ses == 'ses-04': # scene sessions
        enc_type = 'sc'
    
    # ========================================================================
    ### LOAD BEHAVIOURAL DATA:
    # ========================================================================
    # getting timestamps, class labels etc... selection of these things happens below (classifier section)
    
    path_scan1 = os.path.join(path_behav, 'behav_' + sub + '_' + ses + '.pkl')
    
    # import event file
    df_scan1 = pd.read_pickle(path_scan1)
    
    # get run labels from event file
    runs = df_scan1.run.unique()
    # get number of runs in session based on labels
    nruns = len(runs)
    
    # ========================================================================
    ### DEFINE DATA PATHS FOR MASKS, DATA & CONFOUND FILES
    # ========================================================================
    print(f"Loading all masks for {sub} of {ses}")
    # load visual mask task files
    if use_thresholded_masks==1:
        path_mask_VIS_scan1 = os.path.join(path_masks,sub,ses,'ROIs','VIS*T-' + str(mask_threshold) + '*nii')
        path_mask_VIS_scan1 = sorted(glob.glob(path_mask_VIS_scan1), key=lambda f: os.path.basename(f))
    
    # load medial temporal lobe mask task files
        path_mask_MTL_scan1 = os.path.join(path_masks,sub,ses,'ROIs','MTL*T-' + str(mask_threshold) + '*nii')
        path_mask_MTL_scan1 = sorted(glob.glob(path_mask_MTL_scan1), key=lambda f: os.path.basename(f))

    # load hippocampus mask task files
        path_mask_HPC_scan1 = os.path.join(path_masks,sub,ses,'ROIs','HPC*T-' + str(mask_threshold) + '*nii')
        path_mask_HPC_scan1 = sorted(glob.glob(path_mask_HPC_scan1), key=lambda f: os.path.basename(f))
    
    # # load parahippocampus mask task files
    #     path_mask_PHPC_scan1 = os.path.join(path_masks,sub,ses,'ROIs','PHPC*T-' + str(mask_threshold) + '*nii')
    #     path_mask_PHPC_scan1 = sorted(glob.glob(path_mask_PHPC_scan1), key=lambda f: os.path.basename(f))
    
    # # load hippo+parahippocampus mask task files:
    #     path_mask_HPPHC_scan1 = os.path.join(path_masks,sub,ses,'ROIs','HPPHC*T-' + str(mask_threshold) + '*nii')
    #     path_mask_HPPHC_scan1 = sorted(glob.glob(path_mask_HPPHC_scan1), key=lambda f: os.path.basename(f))

    # load entorhinal cortex mask task files
        path_mask_ERC_scan1 = os.path.join(path_masks,sub,ses,'ROIs','ERC*T-' + str(mask_threshold) + '*nii')
        path_mask_ERC_scan1 = sorted(glob.glob(path_mask_ERC_scan1), key=lambda f: os.path.basename(f))

    # load posterior parietal cortex mask task files
        path_mask_PPC_scan1 = os.path.join(path_masks,sub,ses,'ROIs','PPC*T-' + str(mask_threshold) + '*nii')
        path_mask_PPC_scan1 = sorted(glob.glob(path_mask_PPC_scan1), key=lambda f: os.path.basename(f))
    
    # load precuneus mask task files
        path_mask_PREC_scan1 = os.path.join(path_masks,sub,ses,'ROIs','PREC*T-' + str(mask_threshold) + '*nii')
        path_mask_PREC_scan1 = sorted(glob.glob(path_mask_PREC_scan1), key=lambda f: os.path.basename(f))

    # load inferior parietal cortex mask task files
        path_mask_IPL_scan1 = os.path.join(path_masks,sub,ses,'ROIs','IPL*T-' + str(mask_threshold) + '*nii')
        path_mask_IPL_scan1 = sorted(glob.glob(path_mask_IPL_scan1), key=lambda f: os.path.basename(f))

    # # load PFC mask
    #     path_mask_DLPFC_scan1 = os.path.join(path_masks,sub,ses,'ROIs','DLPFC*T-' + str(mask_threshold) + '*nii')
    #     path_mask_DLPFC_scan1 = sorted(glob.glob(path_mask_DLPFC_scan1), key=lambda f: os.path.basename(f))
    
    # load motor cortex mask task files
        path_mask_MOT_scan1 = os.path.join(path_masks,sub,ses,'ROIs','MOT*T-' + str(mask_threshold) + '*nii')
        path_mask_MOT_scan1 = sorted(glob.glob(path_mask_MOT_scan1), key=lambda f: os.path.basename(f))
    
    # # load insula mask task files
    #     path_mask_INS_scan1 = os.path.join(path_masks,sub,ses,'ROIs','INS*T-' + str(mask_threshold) + '*nii')
    #     path_mask_INS_scan1 = sorted(glob.glob(path_mask_INS_scan1), key=lambda f: os.path.basename(f))

    # load whole brain mask task files
        # path_mask_wholebrain_scan1 = os.path.join(path_fmriprep, sub, 'ses-01', 'func','*ses-01*task-gems*T1w*brain_mask.nii.gz')
        path_mask_wholebrain_scan1=os.path.join(path_masks,sub,ses,'ROIs','wholebrain*T-' + str(mask_threshold) + '*nii')
        path_mask_wholebrain_scan1 = sorted(glob.glob(path_mask_wholebrain_scan1), key=lambda f: os.path.basename(f))
    
    # load the functional mri task files
    path_func_task_scan1 = os.path.join(path_fmriprep,sub,ses,'func', '*task-*T1w_desc-preproc_bold.nii.gz')
    path_func_task_scan1 = sorted(glob.glob(path_func_task_scan1), key=lambda f: os.path.basename(f))
    
    # load the anatomical mri file:
    # path_anat_scan1 = os.path.join(path_fmriprep, sub,'anat','%s_desc-preproc_T1w.nii.gz' % sub)
    # path_anat_scan1 = sorted(glob.glob(path_anat_scan1), key=lambda f: os.path.basename(f))
    
    # load the confounds files
    path_confs_task_scan1 = os.path.join(path_fmriprep, sub, ses, 'func','*task-*confounds_timeseries.tsv')
    path_confs_task_scan1 = sorted(glob.glob(path_confs_task_scan1),key=lambda f: os.path.basename(f))
    
    
    # ========================================================================
    ### LOAD EVERYTHING
    # ========================================================================
    
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
    # # load parahippocampus mask:
    # mask_PHPC_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_PHPC_scan1)]
    # mask_PHPC_scan1 = masking.intersect_masks(mask_PHPC_scan1, connected=False, threshold=0)
    # # load hippo+parahippocampus mask:
    # mask_HPPHC_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_HPPHC_scan1)]
    # mask_HPPHC_scan1 = masking.intersect_masks(mask_HPPHC_scan1, connected=False, threshold=0)
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
    # # load dlPFC mask:
    # mask_DLPFC_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_DLPFC_scan1)]
    # mask_DLPFC_scan1 = masking.intersect_masks(mask_DLPFC_scan1, connected=False, threshold=0)
    # load motor mask:
    mask_MOT_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_MOT_scan1)]
    mask_MOT_scan1 = masking.intersect_masks(mask_MOT_scan1, connected=False, threshold=0)
    # # load insula mask:
    # mask_INS_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_INS_scan1)]
    # mask_INS_scan1 = masking.intersect_masks(mask_INS_scan1, connected=False, threshold=0)
    # load wholebrain mask:
    mask_wholebrain_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_wholebrain_scan1)]
    mask_wholebrain_scan1 = masking.intersect_masks(mask_wholebrain_scan1, connected=False)
    
    # define mask dictionary
    masks_scan1 = {
                     'VIS': mask_VIS_scan1, 
                    'MTL': mask_MTL_scan1,
                    'HPC': mask_HPC_scan1, 
                   # 'PHPC': mask_PHPC_scan1,
                   # 'HPPHC': mask_HPPHC_scan1,
                    'ERC': mask_ERC_scan1,
                    'PPC': mask_PPC_scan1,
                    'PREC': mask_PREC_scan1,
                    'IPL': mask_IPL_scan1,
                   # 'DLPFC': mask_DLPFC_scan1,
                    'MOT': mask_MOT_scan1,
                   # 'INS': mask_INS_scan1,
                   'wholebrain': mask_wholebrain_scan1}
    
    # load functional data
    data_task_scan1 = [image.load_img(i) for i in path_func_task_scan1]
    
    # get confound values
    confs_scan1 = [pd.read_csv(i, sep='\t') for i in path_confs_task_scan1]
    
    #### MANUALLY EXCLUDE TOO SHORT RUNS OF BOLD DATA AND CONFOUND FILES OF SUB-02
    if sub == 'sub-02' and enc_type == 'obj' and ses == 'ses-02':
        data_task_scan1 = data_task_scan1[1:] # exclude run 1 (idx 0), as only 161 TRs
        confs_scan1 = confs_scan1[1:]
    elif sub == 'sub-02' and enc_type == 'sc' and ses == 'ses-03':
        data_task_scan1 = data_task_scan1[:7] + data_task_scan1[8:] # exclude run 8 (idx 7), as only 33 TRs
        confs_scan1 = confs_scan1[:7] + confs_scan1[8:]

    
    # set confound variables of interest
    conf_vars = ['global_signal', 'framewise_displacement',
                 'trans_x', 'trans_y', 'trans_z',
                 'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                 'rot_x', 'rot_y', 'rot_z',
                 'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
                 'a_comp_cor_00','a_comp_cor_01', 'a_comp_cor_02','a_comp_cor_03','a_comp_cor_04','a_comp_cor_05']
    
    #sys.exit('LOAD...Stopped before preprocessing')
    
    # ========================================================================
    ### PREPROCESSING
    # ========================================================================
    
    # set kernel size (width of curve) as Full Width at Half Maximum (FWHM) for spatial smoothing
    csmooth = 4

    ### SMOOTH AND DETREND
    # begin looping over masks (whole brain smoothing vs. ROI smoothing)
    for imask, cmask in enumerate(mask_labels):
        print(f"Running {cmask} mask")
       
        # apply whole brain smoothing, then mask data based on a specific anatomical region
        if whole_brain_smooth:
            if not smooth:
                csmooth = None
                print(f"Applying anatomical {cmask} mask without smoothing")
                save_name = 'decod_acc_loc_no_smoothing.pkl'
            else:
                print(f"Running wholebrain smoothing and applying anatomical {cmask} mask")

            # apply mask for each ROI including wholebrain smoothing: get 2d timeseries data (TR number x voxel number)
            data_masked_scan1=[]
            data_masked_scan1 = [masking.apply_mask(i, masks_scan1[cmask], smoothing_fwhm=csmooth) for i in data_task_scan1]
            # # bring masked 2d timeseries data back into 4D space of mask for later comparison
            # unmasked_data_scan1 = []
            # unmasked_data_scan1 = [masking.unmask(i, masks_scan1[cmask]) for i in data_masked_scan1]
        
            # detrend the masked fMRI data separately for each run to increase SNR
            data_masked_detrend_scan1 = [signal.clean(signals=data_masked_scan1[cr],
                                         t_r=TR, detrend=True,
                                         standardize=True,
                                         confounds=confs_scan1[cr][conf_vars].fillna(0))
                                         for cr in range(nruns)]
    
        # mask data first and smooth only masked voxels afterwards
        else:
            print('Applying anatomical ' + cmask + ' mask and smooth ROI voxels afterwards')
            # change save file name
            save_name = 'decod_acc_loc_within_mask.pkl' # saved in sub specific folder

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
            data_masked_scan1 = []
            data_masked_scan1 = [masking.apply_mask(i, masks_scan1[cmask]) for i in ROI_smoothed_data]
        
            # detrend the masked fMRI data separately for each run to increase SNR
            data_masked_detrend_scan1 = [signal.clean(signals=data_masked_scan1[cr],
                                         t_r=TR, detrend=True,
                                         standardize=True,
                                         confounds=confs_scan1[cr][conf_vars].fillna(0))
                                         for cr in range(nruns)]
            
# =============================================================================
#     sys.exit('LOADED & SMOOTHED...Stopped before Cross-validation')
#     t0 = time.time()
#     for imask, cmask in enumerate(mask_labels):
# =============================================================================
        
        # ### PLOT TIMESERIES OF TWO VOXELS FROM X TRs 
        # plt_name = f"2Voxel_timeseries_{sub}_{ses}_{cmask}-mask_{save_name[14:-4]}.png"
        # plt.figure(figsize=(7, 5))
        # plt.plot(data_masked_detrend_scan1[0][:150, :2])
        # plt.xlabel("Time [TRs]", fontsize=16)
        # plt.ylabel("Acivation strength in AU", fontsize=16)
        # plt.xlim(0, 150)
        # plt.title(f"Timeseries of two voxels from {cmask} mask {sub} {ses} {save_name[14:-4]}")
        # plt.subplots_adjust(bottom=0.12, top=0.95, right=0.95, left=0.12)
        # plt.savefig(os.path.join(path_output_plots, plt_name), dpi=130, bbox_inches='tight')#691x525

        
    
        # ========================================================================
        # DEFINE CLASSIFIER PARAMETERS
        # ========================================================================
         
        # select features (e.g. voxels) to a percentile of the highest scores
        feature_selection = SelectPercentile(f_classif, percentile=100)
        
        # create multi-class (multinomial) logistic regression classifier:
            ### 'DOCSTRING' for PARAMETERS:
            # C = C parameter of cost function (fixed to 1)
            # 'lbfgs' algorithm to solve multi-class optimisation problem 
            # using L2 regularisation (ridge regularisation) to prevent coefficients from overfitting
            # training algorithm uses one-vs-rest (OvR) scheme (e.g. leave-one-out cross-validation)
            # classifier take maximum of 10,000 iterations 
            # pattern classification only within (not across!) participants 
        clf = make_pipeline(StandardScaler(), feature_selection,
                            LogisticRegression(C=1., solver='lbfgs', penalty='l2',
                                                 multi_class='ovr', max_iter=10000,
                                                 class_weight='balanced',
                                                 random_state=42))
        
        # ========================================================================
        # TRAIN & TEST ON ENCODING DATA ONLY
        # ========================================================================
        print("Start cross-validation. Training/Testing on encoding trials.")
        # get category labels for showed encoding cue 
        cat_labels = df_scan1.category[(df_scan1.trial_phase == 'enc_cue')] 
        
        # create vector with run number for each showed encoding cue 
        run_vector = df_scan1.run[(df_scan1.trial_phase == 'enc_cue')]
        #run_vector = run_vector.to_numpy() #dtype=object
        run_vector.reset_index(inplace=True, drop=True)
        
        # get timestamp for each encoding cue onset                             
        time_idx = df_scan1.time[(df_scan1.trial_phase == 'enc_cue')]
        
        
        # select TRs of encoding cue presentation of detrended data, account for HRF
        func_data_TRs = pickNprep_TRs(data_masked_detrend_scan1, time_idx, runs, run_vector, HRF=4)
        
        if np.isnan(func_data_TRs).any(): 
            nan_data_tmp, label_tmp, run_tmp = del_NaN_trls(func_data_TRs, cat_labels, run_vector)
            func_data_TRs, cat_labels, run_vector = nan_data_tmp, label_tmp, run_tmp 
        
        if lbl_shuffle:
            cat_labels = np.random.permutation(cat_labels)
            
        ### start leave-one-out Cross-validation (--> train classifier)
        acc_tmp = Classification_TisT(func_data_TRs, cat_labels, runs, run_vector, clf)
        
        # calculate mean of classifier performance of all runs
        decod_acc[imask, ises, 0] = np.mean(acc_tmp)
        
        # ========================================================================
        # TRAIN & TEST ON RETRIEVAL DATA ONLY
        # ========================================================================
        print("Start cross-validation. Training/Testing on retrieval trials.")
        # get category labels for each retrieval trial where button was pressed
        cat_labels = df_scan1.category[(df_scan1.trial_phase == 'ret_resp') & (df_scan1.response == 1)] 
        
        # create vector with run number for each retrieval trial where button was pressed
        run_vector = df_scan1.run[(df_scan1.trial_phase == 'ret_resp') & (df_scan1.response == 1)]
        #run_vector = run_vector.to_numpy() #dtype=object
        run_vector.reset_index(inplace=True, drop=True)
        
        # get timestamp for each for each retrieval trial where button was pressed                        
        time_idx = df_scan1.time[(df_scan1.trial_phase == 'ret_resp') & (df_scan1.response == 1)]
        
        ### exclude nans from time_idx
        time_idx = np.asfarray(time_idx)
        if np.isnan(time_idx).any(): 
            nan_data_tmp, label_tmp, run_tmp = del_NaN_trls(time_idx, cat_labels, run_vector)
            time_idx, cat_labels, run_vector = nan_data_tmp, label_tmp, run_tmp 
        
        # select TRs of encoding cue presentation of detrended data, account for HRF
        func_data_TRs = pickNprep_TRs(data_masked_detrend_scan1, time_idx, runs, run_vector, HRF=4)
        
        if np.isnan(func_data_TRs).any(): 
            nan_data_tmp, label_tmp, run_tmp = del_NaN_trls(func_data_TRs, cat_labels, run_vector)
            func_data_TRs, cat_labels, run_vector = nan_data_tmp, label_tmp, run_tmp 
        
        if lbl_shuffle:
            cat_labels = np.random.permutation(cat_labels)
            
        ### start leave-one-out Cross-validation (--> train classifier)
        acc_tmp = Classification_TisT(func_data_TRs, cat_labels, runs, run_vector, clf)
        
        # calculate mean of classifier performance of all runs
        decod_acc[imask, ises, 1] = np.mean(acc_tmp)
            
        # ========================================================================
        # TRAIN ON ENCODING DATA & TEST ON RETRIEVAL DATA
        # ========================================================================
        print("Start cross-validation. Training on encoding, testing on retrieval trials.")
        ### TRAINING DATA
        # get category labels for encoding cues
        cat_labels_train = df_scan1.category[(df_scan1.trial_phase == 'enc_cue')] 
        
        # create vector with run number for each showed encoding cue 
        run_vector_train = df_scan1.run[(df_scan1.trial_phase == 'enc_cue')]
        #run_vector_train = run_vector_train.to_numpy() #dtype=object
        run_vector_train.reset_index(inplace=True, drop=True)


        # get timestamp for each encoding cue onset                             
        time_idx_train = df_scan1.time[(df_scan1.trial_phase == 'enc_cue')]
        
        # select TRs of encoding cue presentation of detrended data, account for HRF
        func_data_TRs = pickNprep_TRs(data_masked_detrend_scan1, time_idx_train, runs, run_vector_train, HRF=4)
        
        # if runs are too short: delete all trials which are out of bounds
        if np.isnan(func_data_TRs).any(): 
            nan_data_tmp, label_tmp, run_tmp = del_NaN_trls(func_data_TRs, cat_labels_train, run_vector_train)
            func_data_TRs, cat_labels_train, run_vector_train = nan_data_tmp, label_tmp, run_tmp 

        # save training data
        func_data_TRs_train = func_data_TRs 
        
        ### TESTING DATA
        # get category labels for retrieval cues
        cat_labels_test = df_scan1.category[(df_scan1.trial_phase == 'ret_resp') & (df_scan1.response == 1)] 
        
        # run vector for retrieval cues
        run_vector_test = df_scan1.run[(df_scan1.trial_phase == 'ret_resp') & (df_scan1.response == 1)]
        #run_vector_test = run_vector_test.to_numpy() #dtype=object
        run_vector_test.reset_index(inplace=True, drop=True)


        # get timestamps for retrieval cue onsets
        time_idx_test = df_scan1.time[(df_scan1.trial_phase == 'ret_resp') & (df_scan1.response == 1)]
        
        ### exclude nans from time_idx
        time_idx_test = np.asfarray(time_idx_test)
        if np.isnan(time_idx_test).any(): 
            nan_data_tmp, label_tmp, run_tmp = del_NaN_trls(time_idx_test, cat_labels_test, run_vector_test)
            time_idx_test, cat_labels_test, run_vector_test = nan_data_tmp, label_tmp, run_tmp 

        
        # select TRs of retrieval cue presentation, account for HRF
        func_data_TRs = pickNprep_TRs(data_masked_detrend_scan1, time_idx_test, runs, run_vector_test, HRF=4)
        
        # if runs are too short: delete all trials which are out of bounds
        if np.isnan(func_data_TRs).any(): 
            nan_data_tmp, label_tmp, run_tmp = del_NaN_trls(func_data_TRs, cat_labels_test, run_vector_test)
            func_data_TRs, cat_labels_test, run_vector_test = nan_data_tmp, label_tmp, run_tmp 

        # save testing data
        func_data_TRs_test = func_data_TRs 
        
        if lbl_shuffle:
            cat_labels_train = np.random.permutation(cat_labels_train)
            cat_labels_test = np.random.permutation(cat_labels_test)
        ### start leave-one-out Cross-validation (--> train classifier)
        # with train data = encoding trials, test data = retrieval trials
        acc_tmp = Classification_TisNotT(func_data_TRs_train, func_data_TRs_test, cat_labels_train, 
                                         cat_labels_test, runs, run_vector_train, run_vector_test, clf)
        
        # calculate mean of classifier performance of all runs
        decod_acc[imask, ises, 2] = np.mean(acc_tmp)
    
        # #========================================================================
        # # TRAIN & TEST ON JOINT ENC-CUE & RET-CUE DATA
        # #========================================================================
        # print("Start cross-validation. Training/Testing on encoding+retrieval trials.")
        # # get labels 
        # cat_labels = df_scan1.category[(df_scan1.trial_phase == 'enc_cue')
        #                                | (df_scan1.trial_phase == 'ret_resp') & (df_scan1.response == 1)]
        # run_vector = df_scan1.run[(df_scan1.trial_phase == 'enc_cue')
        #                                | (df_scan1.trial_phase == 'ret_resp') & (df_scan1.response == 1)]
        
        # run_vector.reset_index(inplace=True, drop=True)
                               
        # time_idx = df_scan1.time[(df_scan1.trial_phase == 'enc_cue')
        #                                | (df_scan1.trial_phase == 'ret_resp') & (df_scan1.response == 1)]
        
        # ### exclude nans from time_idx
        # time_idx = np.asfarray(time_idx)
        # if np.isnan(time_idx).any(): 
        #     nan_data_tmp, label_tmp, run_tmp = del_NaN_trls(time_idx, cat_labels, run_vector)
        #     time_idx, cat_labels, run_vector = nan_data_tmp, label_tmp, run_tmp 

        # # select TRs of both encoding & retrieval cue presentation, account for HRF
        # func_data_TRs = pickNprep_TRs(data_masked_detrend_scan1, time_idx, runs, run_vector, HRF=4)
        
        # # if runs are too short: delete all trials which are out of bounds
        # if np.isnan(func_data_TRs).any(): 
        #     nan_data_tmp, label_tmp, run_tmp = del_NaN_trls(func_data_TRs, cat_labels, run_vector)
        #     func_data_TRs, cat_labels, run_vector = nan_data_tmp, label_tmp, run_tmp 

        # ### start leave-one-out Cross-validation (--> train classifier)
        # acc_tmp = Classification_TisT(func_data_TRs, cat_labels, runs, run_vector, clf)
        
        # # calculate mean of classifier performance of all runs
        # decod_acc[imask, ises, 3] = np.mean(acc_tmp)
        # test_path2= os.path.join(base_dir, f"decod_1mask{sub}_{ses}_done")
        # os.makedirs(test_path2, exist_ok=True)
                       
    # =============================================================================
    # SAVE DATA IN DATAFRAME AFTER EACH SESSION ITERATION
    # =============================================================================
    
    # save decoding accuracy (classifier performance) for all masks, sessions, testing data cue type
    data_out['acc'] = decod_acc 
    data_out['acc_1dim_label'] = mask_labels
    data_out['acc_2dim_label'] = nses
    data_out['acc_3dim_label'] = cue_type
    
    # create output directory if it doesn't exist already

    #Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = os.path.join(path_output_files, sub, 'permuted')
    os.makedirs(output_dir, exist_ok=True)
        
    data_name = os.path.join(output_dir, save_name)
    
    # save pickle file in sub folder
    with open(data_name, 'wb') as file:
        pickle.dump(data_out, file, protocol = -1)


# stop timer after decoding of all 4 sessions
t1 = time.time()
total = (t1-t0)/60
print('Total run time: ' + str(total) + ' mins') # takes about 15 mins per person. Most time is spent smoothing and applying mask


# # save pickle file in sub folder
# with open(data_name, 'wb') as file:
#     pickle.dump(data_out, file, protocol = -1)


# data_ = pd.Series(data = data_out)
# data_.to_pickle(data_name)

#dat_ = pd.dat_out
#df_out_trimmed.to_pickle(dataname1)
