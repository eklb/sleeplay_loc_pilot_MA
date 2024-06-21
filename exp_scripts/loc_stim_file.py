#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:31:43 2023

@author: kolbe

Creates a stimulus file including predefined stimuli for object and scene localiser 
for n participants.

variables:
    - object encoding stimuli
    - object encoding cues (verbs)
    - object encoding category
    - object retrieval targets
    - object retrieval cues (verbs)
    - ITIs for object encoding phase
    - ITIs for object retrieval phase 
"""
#%% IMPORT ALL PACKAGES
from psychopy import visual, core, gui
from psychopy.hardware import keyboard
import pandas as pd
import numpy as np
import json
import random
import os
from pathlib import Path
import sys
from PIL import Image
import datetime
import pickle
import copy
import matplotlib.pyplot as plt
import csv
#from instructions_loc_ver2 import *

#%% PATHS & FUNCTIONS
# =============================================================================
# DEFINE ALL IMPORTANT PATHS AND VARIABLES
# =============================================================================
#C:\Users\elsak\Documents\GitHub\sleeplay_fMRI_b\tempus\stimuli
#base_dir_win = os.path.join('C:\\', 'Users', 'elsak', 'OneDrive', 'Dokumente', 'GitHub',  'sleeplay_fMRI_b', 'tempus', 'stimuli')
stim_dir = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'GitHub', 
                        'sleeplay', 'behav_online', 'exp_scripts', 'stim')

output_dir = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'MPIB_NeuroCode',
                          'Sleeplay', 'data_analysis', 'MRI_loc_data', 'behav')

wordcue_dir = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'GitHub', 
                        'sleeplay_fMRI_b', 'stimuli')
                          
# =============================================================================
# base_dir = os.path.join('C:\\', 'Users', 'neuroadmin', 'Desktop', 'Marit',
#    'sleeplay_fMRI_b', 'tempus', 'stimuli')
# =============================================================================

obj_dir = os.path.join(stim_dir, 'objects')
cat_list = os.listdir(obj_dir) # list of all object categories from folder
sc_dir = os.path.join(stim_dir, 'scenes')
p_obj_dir = os.path.join(stim_dir, 'practice', 'objects')
p_sc_dir = os.path.join(stim_dir, 'practice', 'scenes')

all_sub = 30
# sub_dict = dict()

# =============================================================================
# DEFINE FUNCTIONS FOR STIM_FILE 
# =============================================================================

def fill_matrix(matrix_name, values, shuffle_idx1, shuffle_idx2):
    """
    Function to create either object or object path matrix from empty matrix.
    Creating temporary list for each block, containing all objects/object paths 
    of a block (20 trials) which is then inserted into columns of matrix. 
    (1 column = 1 block)
    matrix_name = obj_stim_tmp/obj_paths_tmp
    values = all_obj/path_list
    """
    
    # Create temporary lists for each block 
    tmp_list = []
    for i in range(matrix_name.shape[1]):
        tmp_list.append(list())
    
    # values to insert
    tmp_values = copy.deepcopy(values)
    
    # insert one value of each value category into first half of blocks
    for i in range(len(tmp_values)):
        for j in range(len(tmp_list)//2):
            pick = random.choice(tmp_values[i])
            tmp_values[i].remove(pick)
            tmp_list[j].append(pick)
            
    del tmp_values
    
    # shuffle lists with pre-defined shuffle indices             
    for i in range(len(tmp_list)):
        random.seed(shuffle_idx1[i])
        random.shuffle(tmp_list[i])
    
    # insert copy of first half and shuffle with pre-defined shuffle index
    for i in range(len(tmp_list)//2,len(tmp_list)):
        tmp_list[i] = copy.deepcopy(tmp_list[i-len(tmp_list)//2])
        random.seed(shuffle_idx2)
        random.shuffle(tmp_list[i])
    
    for i in range(len(tmp_list)):
        matrix_name[:,i] = tmp_list[i]
    
    # return matrix_name
        

def fill_cues(cue_list, matrix):
# Read in verbs into enc_cues_tmp
    tmp_list = []
    tmp_cues = copy.deepcopy(cue_list)
    
    for i in range(matrix.shape[1]):
        tmp_list.append(tmp_cues[:len(matrix[:,i])])
        del tmp_cues[:len(matrix[:,i])]
        matrix[:, i] = tmp_list[i]
    

def img_resize(img_dir, img_files, size):
    """
    Function to resize all images given the correct directory and image file names.
    img_dir = path_list, img_files = all_obj, size = requested size in pixels 
    as 2-tuple: (width, height)
    """
    
    for i in range(len(img_files)):
        for j in range(len(img_files[i])):
            if os.path.isfile(os.path.join(img_dir[i][j], img_files[i][j])):
                im = Image.open(os.path.join(img_dir[i][j], img_files[i][j]))
                f, e = os.path.splitext(os.path.join(img_dir[i][j], img_files[i][j]))
                imResize = im.resize(size, Image.Resampling.LANCZOS)
                os.remove(os.path.join(img_dir[i][j], img_files[i][j]))
                imResize.save(f+e, 'PNG', quality=90)

def transform_shape(old_mat, new_mat): 
    """
    Function to create matrix of new shape for shorter encoding blocks.
    
    Passed arguments:
        old_mat = obj_stim_tmp, obj_paths_tmp, obj_enc_cues_tmp, ITI_
        new_mat = obj_enc_stim_tmp, obj_enc_paths_tmp, enc_cues_tmp
    """

    for i in range(0, old_mat.shape[1]):
        new_mat[:,2*i] = old_mat[:old_mat.shape[0]//2,i]
        new_mat[:,2*i+1] = old_mat[old_mat.shape[0]//2:old_mat.shape[0],i]
        
def makeITI(meaniti=2.5, rangeiti=[2,7], N=20, maxdiviation=0.01, plot=False):
    """ 
    input:
        meaniti - mean of the ITI
        rangeiti - range of ITI
        N - number of trials
        maxdiviation - how much can the mean of all random samples diviates from the mean? 
        plot - should plot? 
    output: 
        ITIt - vector with ITIs
    """

    worked = False
    while not worked:
        ITIt = np.random.exponential(meaniti-rangeiti[0],N)+rangeiti[0]
        ITIt[ITIt>rangeiti[1]] = rangeiti[1] # push anything above the max to the max
        ITIt[ITIt<rangeiti[0]] = rangeiti[0] # push anything below the min to the min
    #deviation from mean
        worked = (N*meaniti-sum(ITIt))<maxdiviation
    if plot:
        plt.hist(ITIt)
        plt.show()
    return ITIt
   
           
#%% INITIALISE MATRICES FOR ALL SUBJECTS

# =============================================================================
# INITIALISE ALL DICTS
# =============================================================================
obj_enc_stim, obj_enc_paths, obj_enc_cues, obj_enc_cat = dict(), dict(), dict(), dict()
obj_ret_cat, obj_ret_targ, obj_ret_cues = dict(), dict(), dict()
ITI_obj_enc, ITI_obj_ret = dict(), dict()
  
# =============================================================================
# CREATE MATRICES FOR OBJECT ENCODING
# =============================================================================
    #if exp_info["enc_type"] == "obj_enc" and exp_info["sub_ID"] != "sXX":

for isub in range(all_sub):     
    # Create list of sublists containing all object names within each categorie
    all_obj = []
    for i in range(len(cat_list)):
        images = os.listdir(os.path.join(obj_dir, cat_list[i]))
        if '.DS_Store' in images:
            images.remove('.DS_Store')
        for j in range(len(images)):
            obj_name, obj_ext = os.path.splitext(images[j])
            images[j] = obj_name
        all_obj.append(images)
                 
    # Create list of object paths analogous to all_obj list
    path_list = []
    for i in range(len(cat_list)):
        cat_path = os.path.join(stim_dir, obj_dir, cat_list[i])
        path_list.extend([[cat_path]*6])
    
            #img_resize(path_list, all_obj, (508,508))
            
    ### Define all empty numpy arrays for object encoding
    # 10x10 design = half experiment, repition of first half after break
    obj_stim_tmp = np.empty([10, 10], dtype = object)
    obj_paths_tmp = np.empty([10, 10], dtype = object)
    # obj_cat1 = np.empty([10, 10], dtype = object)
    obj_cues_tmp = np.empty([10, 10], dtype = object)
    
    # 5x40 design = 5 cues per block, 4 blocks = 1 run --> 20 trials per run 
    # matrices for stimuli and word cues
    obj_enc_stim_tmp = np.empty([5, 40], dtype = object)
    obj_enc_paths_tmp = np.empty([5, 40], dtype = object)
    obj_enc_cues_tmp = np.empty([5, 40], dtype = object)
    
    # matrices for obj retrieval categories, targets, cues and reaction times
    obj_ret_cat_tmp = np.empty([5, 40], dtype = object)
    obj_targ_tmp = np.empty([5, 40], dtype = object)
    obj_ret_cues_tmp = np.empty([5, 40], dtype = object)
    # obj_rt = np.empty([5, 40], dtype = object)
    
    # # matrices for all time stamps
    # t_obj_enc_fix = np.empty([5, 40], dtype = object)
    # t_obj_enc_cue = np.empty([5, 40], dtype = object)
    # t_obj_ret_fix = np.empty([5, 40], dtype = object)
    # t_obj_ret_cue = np.empty([5, 40], dtype = object)
    
    # arrays for fix cross ITIs for enc and ret separately, jittered per run 
    ITI_obj_enc1 = np.empty([10, 20], dtype = object)
    ITI_obj_ret1 = np.empty([10, 20], dtype = object)
    
    # for 5x40 design of experiment 
    ITI_obj_enc_tmp = np.empty([5, 40], dtype = object)
    ITI_obj_ret_tmp = np.empty([5, 40], dtype = object)
            
    ### create jittered ITIs for fix crosses 
    for i in range(obj_stim_tmp.shape[1]*2):
        tmp = makeITI()
        ITI_obj_enc1[:,i] = tmp[:obj_stim_tmp.shape[0]]
        ITI_obj_ret1[:,i] = tmp[obj_stim_tmp.shape[0]:]
                
    
    ### set shuffle indices and fill matrices 
    
    # set shuffle index to have same for obj and path matrix when calling fill_matrix    
    idx_list = set(list(range(0,100)))
    shuffle_idx1 = random.sample(idx_list, 10)
    shuffle_idx2 = random.randint(0,100)
    
    # Read in obj and corresponding paths into empty arrays (with same shuffle index)
    fill_matrix(obj_stim_tmp, all_obj, shuffle_idx1, shuffle_idx2)
    fill_matrix(obj_paths_tmp, path_list, shuffle_idx1, shuffle_idx2)
    
    # Read in verb stimuli from excel file
    #if exp_info["language"] == "german":
    df_verbs = pd.read_excel(os.path.join(wordcue_dir, 'verbs_german.xlsx'))
    # else:
    #     df_verbs = pd.read_excel(os.path.join(wordcue_dir, 'verbs_english.xlsx'))
    df_verbs.columns = ["verbs"] # name column of table
    verb_list = df_verbs["verbs"].tolist() # store verbs in list
    random.shuffle(verb_list)
    
    # Create verb matrix of half length (2nd half of experiement uses same verbs)
    fill_cues(verb_list, obj_cues_tmp)
    
    # Copy matrices for later concatenation 
    obj_copy = copy.deepcopy(obj_stim_tmp)
    path_copy = copy.deepcopy(obj_paths_tmp)
    verb_copy = copy.deepcopy(obj_cues_tmp)
            
    # Shuffle all belonging matrices with same shuffle index over first dimension, 
    # keeping column structure unchanged, 
    # set seed to random number for each run of script
    seed = np.random.randint(0,100)
    
    rdm_state = np.random.RandomState(seed)
    rdm_state.shuffle(obj_copy)
    rdm_state.seed(seed)
    rdm_state.shuffle(path_copy) 
    rdm_state.seed(seed)
    rdm_state.shuffle(verb_copy)
    
    # Concatenate all matrices to 10x20 shape (-> 20 enc trials per scanner run (10 runs))
    obj_stim_tmp = np.concatenate((obj_stim_tmp, obj_copy), axis=1)
    obj_paths_tmp = np.concatenate((obj_paths_tmp, path_copy), axis=1)
    obj_cues_tmp = np.concatenate((obj_cues_tmp, verb_copy), axis=1)
    
    # =============================================================================
    # # Create object category matrix from path matrix
    # obj_enc_cat1 = copy.deepcopy(obj_paths_tmp)
    # for i in range(obj_paths_tmp.shape[0]):
    #     for j in range(obj_paths_tmp.shape[1]):
    #         obj_enc_cat1[i,j] = os.path.basename(os.path.normpath(obj_enc_cat1[i,j]))
    #      
    # =============================================================================
    
    # Transform shape of 10x20 to 5x40 for shorter encoding blocks
    transform_shape(obj_stim_tmp, obj_enc_stim_tmp)
    transform_shape(obj_paths_tmp, obj_enc_paths_tmp)
    transform_shape(obj_cues_tmp, obj_enc_cues_tmp)
    
    transform_shape(ITI_obj_enc1, ITI_obj_enc_tmp)
    transform_shape(ITI_obj_ret1, ITI_obj_ret_tmp)
    
    # clear workspace, keep only important variables 
    del obj_stim_tmp
    del obj_paths_tmp
    del obj_cues_tmp
       
    #Create object category matrix from path matrix (analogous to encoding)
    obj_enc_cat_tmp = copy.deepcopy(obj_enc_paths_tmp)
    for i in range(obj_enc_cat_tmp.shape[0]):
        for j in range(obj_enc_cat_tmp.shape[1]):
            obj_enc_cat_tmp[i,j] = os.path.basename(os.path.normpath(obj_enc_cat_tmp[i,j]))
         
        # elif exp_info["enc_type"] == "sc_enc" and exp_info["sub_ID"] != "sXX":
     
    # =============================================================================
    # CREATE MATRICES FOR OBJECT RETRIEVAL
    # =============================================================================
     
    #Create retrieval cues and targets 
    obj_ret_cues_tmp = copy.deepcopy(obj_enc_cues_tmp)
    obj_ret_targ_tmp = copy.deepcopy(obj_enc_stim_tmp)
    obj_ret_cat_tmp = copy.deepcopy(obj_enc_cat_tmp)
    
    ### Randomize for now, though pseudo-randomisation necessary!!!
    # use same shuffle index for all matrices, shuffle only rows, keeping column structure 
    seed3 = np.random.randint(0,100)
    
    rdm_state = np.random.RandomState(seed3)
    rdm_state.shuffle(obj_ret_cues_tmp)
    rdm_state.seed(seed3)
    rdm_state.shuffle(obj_ret_targ_tmp)
    rdm_state.seed(seed3)
    rdm_state.shuffle(obj_ret_cat_tmp)
    
    # =============================================================================
    # APPEND TO SUB_DICT
    # =============================================================================
    obj_enc_stim[f"{isub}"] = obj_enc_stim_tmp
    obj_enc_paths[f"{isub}"] = obj_enc_paths_tmp
    obj_enc_cues[f"{isub}"] = obj_enc_cues_tmp
    obj_enc_cat[f"{isub}"] = obj_enc_cat_tmp
    obj_ret_cat[f"{isub}"] = obj_ret_cat_tmp
    obj_ret_targ[f"{isub}"] = obj_ret_targ_tmp
    obj_ret_cues[f"{isub}"] = obj_ret_cues_tmp
    ITI_obj_enc[f"{isub}"] = ITI_obj_enc_tmp
    ITI_obj_ret[f"{isub}"] = ITI_obj_ret_tmp
    
    

#%%   
# save all to dict structure 

# initialise sub_dict
sub_dict = {f"{isub}": isub+1 for isub in range(all_sub)}

main_dict = {}
main_dict["sub_IDs"] = sub_dict
# main_dict["loc_sess"] = [loc_date]
# main_dict["loc_sess"].append(loc_date)
main_dict["obj_enc_stim"] = obj_enc_stim
main_dict["obj_enc_paths"] = obj_enc_paths
   # main_dict["obj_img_ext"] = obj_ext
main_dict["obj_enc_cat"] = obj_enc_cat
main_dict["obj_ret_cat"] = obj_ret_cat
main_dict["obj_enc_cues"] = obj_enc_cues
main_dict["obj_ret_cues"] = obj_ret_cues
main_dict["obj_ret_targ"] = obj_ret_targ
# main_dict["obj_ret_rt"] = obj_rt
# main_dict["t_obj_enc_fix"] = t_obj_enc_fix
# main_dict["t_obj_enc"] = t_obj_enc
# main_dict["t_obj_ret_fix"] = t_obj_ret_fix
# main_dict["t_obj_ret_cue"] = t_obj_ret_cue
main_dict["ITI_obj_enc"] = ITI_obj_enc
main_dict["ITI_obj_ret"] = ITI_obj_ret

# main_dict["enc_sc"] = sc_matrix
# main_dict["sc_paths"] = sc_paths
#    # main_dict["sc_img_ext"] = sc_ext
# main_dict["enc_sc_adj"] = sc_adj
# main_dict["sc_ret_cues"] = sc_cues
# main_dict["sc_ret_targ"] = sc_targ
# main_dict["sc_ret_rt"] = sc_rt
# main_dict["t_sc_enc_fix"] = t_sc_enc_fix
# main_dict["t_sc_enc"] = t_sc_enc
# main_dict["t_sc_ret_fix"] = t_sc_ret_fix
# main_dict["t_sc_ret_cue"] = t_sc_ret_cue
# main_dict["ITI_sc_enc"] = ITI_sc_enc
# main_dict["ITI_sc_ret"] = ITI_sc_ret





#%% 

# =============================================================================
# CREATE MATRICES FOR SCENE ENCODING
# =============================================================================
# Create list of all scene images     
sc_stim = os.listdir(sc_dir)
if '.DS_Store' in sc_stim:
    sc_stim.remove('.DS_Store')
for i in range(len(sc_stim)):
    sc_name, sc_ext = os.path.splitext(sc_stim[i])
    sc_stim[i] = sc_name
        

# =============================================================================
# # Create list of sc paths with len(sc_stim) for image resizing
# sc_paths = []
# sc_paths.extend([sc_dir]*len(sc_stim))
# #img_resize(sc_paths, sc_stim, (700,420)) #size-presentation 508x304,8
# =============================================================================

# Define all empty matrices for scene encoding
sc_stim_tmp = np.empty([10, 10], dtype = object)
sc_paths_tmp = np.empty([10, 10], dtype = object)
sc_cues_tmp = np.empty([10, 10], dtype = object)

sc_enc_stim_tmp = np.empty([5, 40], dtype = object)
sc_paths = np.empty([5, 40], dtype = object)
sc_adj = np.empty([5, 40], dtype = object)

sc_targ = np.empty([5, 40], dtype = object)
sc_cues = np.empty([5, 40], dtype = object)
sc_rt = np.empty([5, 40], dtype = object)

# t_sc_enc_fix = np.empty([5, 40], dtype = object)
# t_sc_enc_cue = np.empty([5, 40], dtype = object)
# t_sc_ret_fix = np.empty([5, 40], dtype = object)
# t_sc_ret_cue = np.empty([5, 40], dtype = object)

# arrays for ITIs
ITI_sc_enc1 = np.empty([10, 20], dtype = object)
ITI_sc_ret1 = np.empty([10, 20], dtype = object)

ITI_sc_enc = np.empty([5, 40], dtype = object)
ITI_sc_ret = np.empty([5, 40], dtype = object)

#create ITIs
for i in range(sc_stim_tmp.shape[1]*2):
    tmp = makeITI()
    ITI_sc_enc1[:,i] = tmp[:sc_stim_tmp.shape[1]]
    ITI_sc_ret1[:,i] = tmp[sc_stim_tmp.shape[1]:]
            
# Read in sc stim and sc paths data into empty matrices
tmp_list = []
for i in range(sc_stim_tmp.shape[1]):
    sc_block = copy.deepcopy(sc_stim)
    random.shuffle(sc_block)
    tmp_list.append(sc_block)

for i in range(len(tmp_list)):
    sc_stim_tmp[:,i] = tmp_list[i]
    #sc_paths1[:,i] = sc_paths
        
# Read in sc adjectives
# if exp_info["language"] == "german":
df_adj = pd.read_excel(os.path.join(wordcue_dir, 'adjectives_german.xlsx'))
# else:
#     df3 = pd.read_excel(os.path.join(wordcue_dir, 'adjectives_english.xlsx'))
df_adj.columns = ["adjectives"] 
adj_list = df_adj["adjectives"].tolist()
random.shuffle(adj_list)

#Create adj matrix of half length
fill_cues(adj_list, sc_adj1)

# Deepcopy sc matrices for later concatenation of copy to existing matrix 
sc_copy = copy.deepcopy(sc_stim_tmp)
adj_copy = copy.deepcopy(sc_adj1)

# Shuffle all belonging matrices with same shuffle index over first dimension, 
# keeping column structure, set seed to random number for each run of script
seed2 = np.random.randint(0,100)

rdm_state = np.random.RandomState(seed2)
rdm_state.shuffle(sc_copy)
rdm_state.seed(seed2)
rdm_state.shuffle(adj_copy) 

# Concatenate all matrices to 10x20 shape
sc_stim_tmp = np.concatenate((sc_stim_tmp, sc_copy), axis=1)
sc_adj1 = np.concatenate((sc_adj1, adj_copy), axis=1)
sc_paths_tmp = np.concatenate((sc_paths_tmp, sc_paths1), axis=1)

#Create new matrices(5,40) for shorter encoding
transform_shape(sc_stim_tmp, sc_enc_stim_tmp)
transform_shape(sc_paths1, sc_paths)
transform_shape(sc_adj1, sc_adj)

transform_shape(ITI_sc_enc1, ITI_sc_enc)
transform_shape(ITI_sc_ret1, ITI_sc_ret)

# set date    
date = datetime.datetime.today()
date = date.strftime("%d/%m/%Y/%H:%M")

#Create main dict containing all important values for each subject
loc_date_obj = "_".join(['obj_enc', date])
loc_date_sc = "_".join(['sc_enc', date])

    
    # =============================================================================
    # if os.path.exists(file_path) and exp_info['block_no'] != 'block_1':
    #     with open(file_path, "rb") as file:
    #         stim_file_dict = pickle.load(file)
    # =============================================================================
    #if exp_info["block_no"] == "block_1":     
stim_file_dict = {}
stim_file_dict["sub_ID"] = 'sub-01'
stim_file_dict["loc_sess"] = [loc_date_obj]
        
    # else:
    #     with open(file_path, "rb") as file:
    #         stim_file_dict = pickle.load(file)
       
stim_file_dict["loc_sess"].append(loc_date_sc)
            
    # if exp_info["enc_type"] == "obj_enc":
stim_file_dict["obj_enc_cues"] = enc_stim_tmp
# stim_file_dict["enc_paths_tmp"] = enc_paths_tmp
# stim_file_dict["obj_img_ext"] = obj_ext
stim_file_dict["obj_ret_cat"] = obj_cat
stim_file_dict["obj_enc_verbs"] = enc_cues_tmp
stim_file_dict["obj_ret_cues"] = obj_cues
stim_file_dict["obj_ret_targ"] = obj_targ
stim_file_dict["obj_ret_rt"] = obj_rt
# stim_file_dict["t_obj_enc_fix"] = t_obj_enc_fix
# stim_file_dict["t_obj_enc_cue"] = t_obj_enc_cue
# stim_file_dict["t_obj_ret_fix"] = t_obj_ret_fix
# stim_file_dict["t_obj_ret_cue"] = t_obj_ret_cue
stim_file_dict["ITI_obj_enc"] = ITI_obj_enc
stim_file_dict["ITI_obj_ret"] = ITI_obj_ret
    
    # else:
stim_file_dict["sc_enc_cues"] = sc_matrix
# stim_file_dict["sc_paths"] = sc_paths
# stim_file_dict["sc_img_ext"] = sc_ext
stim_file_dict["sc_enc_adj"] = sc_adj
stim_file_dict["sc_ret_cues"] = sc_cues
stim_file_dict["sc_ret_targ"] = sc_targ
stim_file_dict["sc_ret_rt"] = sc_rt
stim_file_dict["t_sc_enc_fix"] = t_sc_enc_fix
stim_file_dict["t_sc_enc_cue"] = t_sc_enc_cue
stim_file_dict["t_sc_ret_fix"] = t_sc_ret_fix
stim_file_dict["t_sc_ret_cue"] = t_sc_ret_cue
stim_file_dict["ITI_sc_enc"] = ITI_sc_enc
stim_file_dict["ITI_sc_ret"] = ITI_sc_ret


# df_data = pd.DataFrame(stim_file_dict)


# csv_file = 'example_dict'
# with open(csv_file, 'w') as csvfile:
#         writer = csv.DictWriter(csvfile)#fieldnames=csv_columns
#         writer.writeheader()
#         for data in stim_file_dict:
#             writer.writerow(data)
        
#sys.exit("Running of script has been interrupted")

# =============================================================================
#     
# # save main dict to pickle file
# with open(file_path, "wb") as file:
#     pickle.dump(stim_file_dict, file, pickle.HIGHEST_PROTOCOL)
#     
# # backup
# with open(file_path2, "wb") as file:
#     pickle.dump(stim_file_dict, file, pickle.HIGHEST_PROTOCOL)
# 
# # =============================================================================
# =============================================================================
# if exp_info["loc_sess"] == "loc1":
#     with open(file_path2, "wb") as file:
#         pickle.dump(stim_file_dict, file, pickle.HIGHEST_PROTOCOL)
# =============================================================================
