#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:46:07 2023

@author: kolbe
"""

import sys
import os
import glob
#matplotlib inline
import numpy as np
import nibabel as nb
from nilearn import image, masking#, signal, plotting
#from nilearn.plotting import plot_img
from nilearn.decoding import SearchLight
from nilearn.image import new_img_like #,resample_to_img, math_img
from nilearn.maskers import NiftiMasker
# from scipy.ndimage import binary_dilation
from scipy.signal import detrend
from scipy.stats import zscore
# import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import make_pipeline
import pandas as pd
import time
# from tqdm import tqdm
#import nipype.interfaces.fsl as fsl


# start timer
t0 = time.time()

# =============================================================================
# ### DEFINE IMPORTANT PATHS:
# =============================================================================

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
    
# if running on cluster, get sub ID from bash runner file
if run_clust:
    # get subject id:
    sub = 'sub-%s' % sys.argv[1]
    ses = 'ses-%s' % sys.argv[2]
    n_rep = str(sys.argv[3])
# if running locally, define sub ID here
else:
    sub = 'sub-02'
    ses = 'ses-02'
    
behav_dir = os.path.join(base_dir, 'behav', 'decod_prep_data') # path to behavioural data 
data_dir = os.path.join(path_project, fmri_prep_suffix)
save_dir = os.path.join(path_project, 'searchlight')
save_dir_plots = os.path.join(save_dir, 'plots')

os.makedirs(save_dir_plots, exist_ok=True)


### DEFINE CONDITIONS, SPACE AND WHETHER DATA SHOULD BE SMOOTHED
cue_types = ['enc'] #'ret' # , 'enc&ret', 'enc2ret' # where classifiers are trained/tested on 
TR = 1.25
MNI = True
lbl_shuffle = True #True for label shuffeling with X repetions, False for real labels

# set either csmooth to None for no spatial smoothing
# or to a kernel size as Full Width at Half Maximum (FWHM) for spatial smoothing (here 4cm)
csmooth = None # 4
smooth_suffix = 'nosmooth' # 'smooth'


enc_type = 'obj' # required to load behaviour
if ses == 'ses-03' or ses == 'ses-04': # scene sessions
    enc_type = 'sc'


# ========================================================================
### LOAD BEHAVIOURAL DATA:
# ========================================================================
# getting timestamps, class labels etc... selection of these things happens below (classifier section)

behav_data = os.path.join(behav_dir, 'behav_' + sub + '_' + ses + '.pkl')

# import event file
behav_df = pd.read_pickle(behav_data)

# get run labels from event file
runs = behav_df.run.unique()
# get number of runs in session based on labels
nruns = len(runs)

#get labels from event file
cond_labels = np.unique(behav_df.category)

#%% Define all other functions for preparation of functional data

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

def pickNprep_TRs(func_data, time_idx, cat_labels, runs, run_vector, HRF):    
    ### get relevant TRs from preprocessed data
    func_data_TRs, corr_labels, chunks = list(), list(), list()

    # find closest TR of each timestamp of cue presentation
    TR_idx = np.empty(len(time_idx))
    for i, t in enumerate(time_idx):
        TR_idx[i] = find_closest_TR(t, 0, 1000, TR)
    
    # add time shift of 4 TRs to tr_idx (for HRF response)
    #global TR_idx
    TR_idx = TR_idx + HRF
    
    ## if run too short, delete TR trials which are out of bound
   # del_TRs = 
    # TR_idx = np.delete(TR_idx, np.where(TR_idx>=data_tmp.shape[3]))
    
    # extract label indices
    
    # select data according to TRs of each run
    for irun in range(0,len(runs)):
        data_tmp = func_data[irun]
        idx_tmp = TR_idx[run_vector == irun+1].astype(int) # array with TR's per run
        label_tmp = cat_labels[run_vector == irun+1]

        # ## if run too short, delete TR trials which are out of bound
        del_TRs = idx_tmp>=data_tmp.shape[3]
        count_del_TRs = np.count_nonzero(del_TRs)
        if count_del_TRs != 0:
            print(f"Run {irun+1} of {ses}: A total of {count_del_TRs} Trials are out of bound. \nDelete...")

        # boolean array with True values for TR's to be deleted 
        idx_tmp = np.delete(idx_tmp, np.where(del_TRs))
        # also delete corresponding labels 
        label_tmp = np.delete(label_tmp, np.where(del_TRs))
        # create run vector of new length
        run_tmp = np.full(len(idx_tmp), irun+1)
        # select TR's of nifti data
        sel_TRs = image.index_img(data_tmp, idx_tmp)
        
        #append selected TRs, labels, chunks (aka run numbers) to list
        func_data_TRs.append(sel_TRs)
        corr_labels.append(label_tmp)
        chunks.append(run_tmp)


    # combine (concatenate) the selected volumes, corresponding labels and chunks of all runs:
    func_data_TRs = image.concat_imgs(func_data_TRs)
    corr_labels, chunks = np.concatenate(corr_labels), np.concatenate(chunks)
    
    return func_data_TRs, corr_labels, chunks

def del_NaN_trls(nan_data, cat_labels, run_vector):
    '''
    Delete all trials of functional data, class labels and run vector from 
    shorter scan runs which are out of bounds. 
    '''
    # check for NaNs in functional data and delete trials (and corresponding labels/trials of run vector)
    # global nan_data_tmp, label_tmp, run_tmp
    # dtime_idx, dcat_labels, drun_vector = np.empty(nan_data.shape), np.empty(nan_data.shape), np.empty(nan_data.shape)
    if nan_data.ndim == 1:
        nan_idc = np.isnan(nan_data)
    else:
        nan_idc = np.isnan(nan_data[:, 0])
    nan_idc = nan_idc.flatten('F')
    print(f"Deleting {np.count_nonzero(nan_idc)} nan trials")
    # dtime_idx = np.delete(nan_data, nan_idc, axis=0)
    # dcat_labels = np.delete(cat_labels, nan_idc)
    # drun_vector = run_vector.drop(nan_idc)
    nan_data = nan_data[~nan_idc]
    cat_labels = cat_labels[~nan_idc]
    run_vector = run_vector[~nan_idc]
    return nan_data, cat_labels, run_vector

#%% CREATE PATH LISTS FOR FUNC DATA, MASKS, CONFOUNDS SORTED FOR ALL RUNS FOR ALL SUBS

#loop over all subjects and sessions in .sh script for parralising

for cue_type in cue_types:
    # functional MRI files 
    if MNI:
        path_func_scan = os.path.join(data_dir, sub, ses, 'func', '*task-*MNI152NLin6Asym_desc-preproc_bold.nii.gz')
        path_func_scan = sorted(glob.glob(path_func_scan), key=lambda f: os.path.basename(f)) # = paths
        
        # confounds files
        path_confounds = os.path.join(data_dir, sub, ses, 'func','*task-*confounds_timeseries.tsv')
        path_confounds = sorted(glob.glob(path_confounds),key=lambda f: os.path.basename(f))
        
        # anatomical, functional mask
        path_anat_mask = os.path.join(data_dir, sub, ses, 'func', '*task-*MNI152NLin6Asym_desc-aparcaseg_dseg.nii.gz')
        path_anat_mask = sorted(glob.glob(path_anat_mask),key=lambda f: os.path.basename(f))
        path_func_mask = os.path.join(data_dir, sub, ses, 'func', '*task-*MNI152NLin6Asym_desc-brain_mask.nii.gz')
        path_func_mask = sorted(glob.glob(path_func_mask),key=lambda f: os.path.basename(f))
        
    else:
        path_func_scan = os.path.join(data_dir, sub, ses, 'func', '*task-*T1w_desc-preproc_bold.nii.gz')
        path_func_scan = sorted(glob.glob(path_func_scan), key=lambda f: os.path.basename(f)) # = paths
    
        # confounds files
        path_confounds = os.path.join(data_dir, sub, ses, 'func','*task-*confounds_timeseries.tsv')
        path_confounds = sorted(glob.glob(path_confounds),key=lambda f: os.path.basename(f))
        
        # anatomical, functional mask
        path_anat_mask = os.path.join(data_dir, sub, ses, 'func', '*task-*T1w_desc-aparcaseg_dseg.nii.gz')
        path_anat_mask = sorted(glob.glob(path_anat_mask),key=lambda f: os.path.basename(f))
        path_func_mask = os.path.join(data_dir, sub, ses, 'func', '*task-*T1w_desc-brain_mask.nii.gz')
        path_func_mask = sorted(glob.glob(path_func_mask),key=lambda f: os.path.basename(f))

#%% LOAD AND PREPROCESS DATA

    # =====================================================================
    # LOAD FUNCTIONAL DATA AND CONFOUND VARIABLES
    # =====================================================================
    
    # Load func brain mask for each run in which searchlight should be conducted (for whole brain analysis)
    func_mask = [image.load_img(i) for i in path_func_mask]
    
    # Load anat brain mask for each run for plotting
    # anat_mask = [image.load_img(i) for i in path_anat_mask]
    
    # Compute intersection of functional masks from all runs
    int_func_mask = masking.intersect_masks(func_mask)
    
    #func_img, func, labels =  prepare_func_data(paths, logs, conds2decode, confounds,img_mask)
    
    # load func data nifti files of all runs
    func_data = [image.load_img(i) for i in path_func_scan]
    #img_func = image.load_img(path)#nb.load(path)
    
    # get confound values
    conf_data = [pd.read_csv(i, sep='\t') for i in path_confounds]
	
    #### MANUALLY EXCLUDE TOO SHORT RUNS OF BOLD DATA AND CONFOUND FILES OF SUB-02
    if sub == 'sub-02' and enc_type == 'obj' and ses == 'ses-02':
        func_data = func_data[1:] # exclude run 1 (idx 0), as only 161 TRs
        conf_data = conf_data[1:]
    elif sub == 'sub-02' and enc_type == 'sc' and ses == 'ses-03':
        func_data = func_data[:7] + func_data[8:] # exclude run 8 (idx 7), as only 33 TRs
        conf_data = conf_data[:7] + conf_data[8:]
    
    
    # set confound variables of interest
    conf_vars = ['global_signal', 'framewise_displacement',
                 'trans_x', 'trans_y', 'trans_z',
                 'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                 'rot_x', 'rot_y', 'rot_z',
                 'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
                 'a_comp_cor_00','a_comp_cor_01', 'a_comp_cor_02','a_comp_cor_03','a_comp_cor_04','a_comp_cor_05']
    
    # test_path= os.path.join(base_dir, f"data_loaded_{sub}_{ses}_done")
    # os.makedirs(test_path, exist_ok=True)
    
    # =====================================================================
    # PREPROCESS: smooth, detrend and zscore data, remove confounds
    # =====================================================================
   
    # func_smooth_ = [image.smooth_img(i, fwhm=csmooth) for i in func_data]
    
    # # Extract whole-brain time series data without pre-processing for plotting raw
    # masker_raw = NiftiMasker(mask_img=func_mask[0], 
    #                           smoothing_fwhm=csmooth,
    #                           standardize=False,
    #                           detrend=False,
    #                           standardize_confounds=False,
    #                           t_r=TR)
    
    # raw_timeseries = masker_raw.fit_transform(func_data[0])
    
    
    # Create a NiftiMasker for whole brain (based on functional mask) with parameters for preprocessing 
    masker = [NiftiMasker(mask_img=i, 
                         smoothing_fwhm=csmooth,
                         standardize="zscore_sample",
                         detrend=True,
                         t_r=TR) 
              for i in func_mask]
    
    del func_mask
    
    # Extract time series and detrend, standardise and remove confounds from fMRI data
    # separately for each run to increase SNR
    func_preproc = [masker[i].fit_transform(func_data[i], 
                                            confounds=conf_data[i][conf_vars].fillna(0))
                    for i in range(nruns)]
    
    del func_data
    
    # bring time_series data back into nifti format
    func_preproc4D = [masker[i].inverse_transform(func_preproc[i]) for i in range(nruns)]
    
    del func_preproc

    
#%% BEGIN SEARCHLIGHT

    # =====================================================================
    # DEFINE CLASSIFIER AND SEARCHLIGHT OBJECT PARAMETERS
    # =====================================================================
      
    # select features (e.g. voxels) to a percentile of the highest scores
    feature_selection = SelectPercentile(f_classif, percentile=100)
     
    # specify classfier: create multi-class (multinomial) logistic regression classifier
    clf = make_pipeline(StandardScaler(), 
                        feature_selection,
                        LogisticRegression(C=1., 
                                           solver='lbfgs', 
                                           penalty='l2',
                                           multi_class='ovr', 
                                           max_iter=10000,
                                           class_weight='balanced',
                                           random_state=42))
    
    # Specify the radius of the searchlight sphere that will scan the volume
    # (the bigger the longer the computation)
    sphere_r = 4  # in mm
    
    # Create searchlight object
    sl = SearchLight(int_func_mask,
                     # process_mask_img=int_func_mask,
                     radius=sphere_r,
                     estimator=clf,
                     cv=LeaveOneGroupOut(),
                     n_jobs=-1, # -1 = ‘all CPUs’
                     verbose=1)
    # ========================================================================
    # TRAIN & TEST ON ENCODING DATA ONLY
    # ========================================================================
    if cue_type == 'enc':
        print("Start cross-validation. Training/Testing on encoding trials.")
        # get category labels for showed encoding cue 
        cat_labels = behav_df.category[(behav_df.trial_phase == 'enc_cue')] 
        
        # create vector with run number for each showed encoding cue 
        run_vector = behav_df.run[(behav_df.trial_phase == 'enc_cue')]
        #run_vector = run_vector.to_numpy() #dtype=object
        run_vector.reset_index(inplace=True, drop=True)
        
        # get timestamp for each encoding cue onset                             
        time_idx = behav_df.time[(behav_df.trial_phase == 'enc_cue')]
        
    # ========================================================================
    # TRAIN & TEST ON RETRIEVAL DATA ONLY
    # ========================================================================
    elif cue_type == 'ret':
        print("Start cross-validation. Training/Testing on retrieval trials.")
        # get category labels for each retrieval trial where button was pressed
        cat_labels = behav_df.category[(behav_df.trial_phase == 'ret_resp') & (behav_df.response == 1)] 
        
        # create vector with run number for each retrieval trial where button was pressed
        run_vector = behav_df.run[(behav_df.trial_phase == 'ret_resp') & (behav_df.response == 1)]
        #run_vector = run_vector.to_numpy() #dtype=object
        run_vector.reset_index(inplace=True, drop=True)
        
        # get timestamp for each for each retrieval trial where button was pressed                        
        time_idx = behav_df.time[(behav_df.trial_phase == 'ret_resp') & (behav_df.response == 1)]
       
    # =====================================================================
    # CLASSIFICATION: TRAIN & TEST ON ENCODING + RETRIEVAL DATA
    # =====================================================================
    # elif cue_type == 'enc+ret':
    else:   
        print("Start cross-validation. Training/Testing on encoding+retrieval trials.")
        
        ### get labels, run_nr, time_stamps which correspond to enc+ret cue presentation only 
        # (->ret cues only if button was pressed)
        cat_labels = behav_df.category[(behav_df.trial_phase == 'enc_cue')
                                       | (behav_df.trial_phase == 'ret_resp') & (behav_df.response == 1)]
        run_vector = behav_df.run[(behav_df.trial_phase == 'enc_cue')
                                       | (behav_df.trial_phase == 'ret_resp') & (behav_df.response == 1)]
        
        run_vector.reset_index(inplace=True, drop=True)
                               
        time_idx = behav_df.time[(behav_df.trial_phase == 'enc_cue')
                                       | (behav_df.trial_phase == 'ret_resp') & (behav_df.response == 1)]
    
    # exclude nans from time_idx, also delete corresponding labels/run_nr 
    time_idx = np.asfarray(time_idx)
    if np.isnan(time_idx).any(): 
        time_idx, cat_labels, run_vector = del_NaN_trls(time_idx, cat_labels, run_vector)

    # select corresponding volumes/TRs of both encoding & retrieval cue presentation, account for HRF
    func_data_TRs, corr_labels, chunks = pickNprep_TRs(func_preproc4D, time_idx, cat_labels, runs, run_vector, HRF=4)
    
    # test_path2= os.path.join(base_dir, f"fitting_SL_{sub}_{ses}")
    # os.makedirs(test_path2, exist_ok=True)
    
    if lbl_shuffle:
        corr_labels = np.random.permutation(corr_labels)
    
    # Run the searchlight algorithm
    print("Fitting Searchlight...")
    sl.fit(func_data_TRs, corr_labels, groups=chunks)
    
    t1 = time.time()
    total = ((t1-t0)/60)//60
    print(f"Total run time {sub} {ses} {cue_type}: " + str(total) + ' hours') 


#%% SAVE OUTPUT MAPS

    # bring searchlight output back into nifti format
    searchlight_img = new_img_like(int_func_mask, sl.scores_)
    
    # Save the result nii img
    sl_filetype = '.nii'
    map_name =  f"Searchlight-AccMap_{sub}_{ses}_{cue_type}_{smooth_suffix}"
    sl_map_savename = map_name 
    
    save_dir_files = os.path.join(save_dir, 'files', f"{sub}", f"{ses}", f"{cue_type}", 'observed')
    os.makedirs(save_dir_files, exist_ok=True)
    
    if MNI:
        sl_map_savename =  map_name + '_MNI'
        
    if lbl_shuffle:
        sl_map_savename =  sl_map_savename + f"_permuted{n_rep}"
        
        save_dir_files = os.path.join(save_dir, 'files', f"{sub}", f"{ses}", f"{cue_type}", 'permuted')
        os.makedirs(save_dir_files, exist_ok=True)

    nb.save(searchlight_img, os.path.join(save_dir_files, sl_map_savename + sl_filetype))


#%% ...PLOT SEARCH LIGHT SCORES_ (in the process)

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


