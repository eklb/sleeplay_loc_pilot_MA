#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sleeplay. berlin_loc_v2
compare post-scan test with in scan retrieval
"""

import os
import numpy as np
from scipy import stats
import sys
#import pandas as pd
from matplotlib import pyplot as plt
from cycler import cycler
#import matplotlib.pyplot as plt
import pickle

script_dir = os.path.join('C:\\', 'Users', 'elsak', 'OneDrive', 'Dokumente', 'GitHub_c3', 
                          'b_loc_v2')

base_dir = os.path.join('C:\\', 'Users', 'elsak', 'OneDrive', 'Dokumente', 'MPIB_NeuroCode', 
                        'Sleeplay', 'data_analysis') 

sub_dir = os.path.join(base_dir, 'behav_loc_data', 'pilots') #where is the data

save_dir = os.path.join(base_dir, 'behav_loc_data', 'analyses') # where data will be saved 

stim_dir = os.path.join(script_dir, 'exp_scripts', 'stimuli','images')
sc_dir = 'scenes'
obj_dir = 'obj_all' # folder with all object images without categories as folder structure

sub = ['s01', 's02', 's03'] #'s01', 's02', 's03', 's04'
ses = ['ses01'] #'ses02'
obj_sc = ['obj', 'sc'] #'sc'

match = np.empty([len(sub), len(obj_sc)]) # sub x task 
match[:] = np.nan

#loop through subjects, enc type and sessions
for isub, subID in enumerate(sub):
    
    for itask, etask in enumerate(obj_sc):
         
        # getting encoding/retrieval data
        for ises, eses in enumerate(ses):
            data_dir = os.path.join(sub_dir, sub[isub], 'behav', obj_sc[itask], ses[ises])
            #print(data_dir)
        
            # load memory data (with scored ret_dict appended)
            for ifile in os.listdir(data_dir):
                if ifile[-15:] == 'memory_data.pkl':
                    mem_fname = ifile
                    
                    with open(os.path.join(save_dir, mem_fname), "rb") as file:
                        mem_file = pickle.load(file)
                    
                    # create empty array for match of ret cues with target input
                    match_ret = np.zeros([len(mem_file['ret_dict']['corr_inp'])])
                    
                    # iterating through ret cues from retrieval dict and get indices 
                    # of same cues in main dict matrix
                    miss_cue = 0
                    # count empty cues
                    for itrl in range(len(mem_file['ret_dict']['corr_inp'])):
                        ret_cue = mem_file['ret_dict'][etask + '_ret_cues'][itrl]
                        cue_idx = np.array(np.where(mem_file[etask + '_ret_cues'] == ret_cue))
                        if cue_idx.size == 0:
                            miss_cue += 1
                            pass                        
                        else:
                            # columns represent indices of ret cue
                            # sort 2nd row and keep overall column structure to 
                            # pick last idx pair of task
                            cue_idx = cue_idx[:, cue_idx[1,:].argsort()]
                            # saving only idx of 2nd cue repetition 
                            cue_idx = (cue_idx[0,-1], cue_idx[1][-1])
                            
                            # get reaction time of last retrieval cue
                            rt_cue = mem_file[etask + '_ret_rt'][cue_idx]
                            
                            # get response during test
                            corr_inp = mem_file['ret_dict']['corr_inp'][itrl]
                            
                            hit, miss, fls_pos, tru_neg = 0, 0, 0, 0
                            if ~np.isnan(rt_cue) and corr_inp == 1:
                                match_ret[itrl] = 1 # hits (true positive)
                                #hit += 1
                            elif np.isnan(rt_cue) and corr_inp == 1:
                                match_ret[itrl] = 2 # misses (false negative)
                                #miss += 1
                            elif ~np.isnan(rt_cue) and corr_inp == 0:
                                match_ret[itrl] = 3 # false alarm (false positive)
                                #fls_pos += 1
                            elif np.isnan(rt_cue) and corr_inp == 0:
                                match_ret[itrl] = 4 # correct rejection (true negative)
                                #tru_neg += 1
                            
                            # count absolute responses
                            for resp in match_ret:
                                if resp == 1:
                                    hit +=1
                                elif resp == 2:
                                    miss +=1
                                elif resp == 3:
                                    fls_pos +=1
                                elif resp ==4:
                                    tru_neg +=1
                            
                    # create confusion matrix with values in percent 
                    con_mat = np.zeros((2,2))
                    con_mat[0][0] = round(hit/(len(match_ret)-miss_cue), 4)
                    con_mat[0][1] = round(miss/(len(match_ret)-miss_cue), 4)
                    con_mat[1][0] = round(fls_pos/(len(match_ret)-miss_cue), 4)
                    con_mat[1][1] = round(tru_neg/(len(match_ret)-miss_cue), 4)
                    
# =============================================================================
#                     # plot matrix for each subject separately 
#                     plt_path = os.path.join(data_dir, (mem_fname.replace(mem_fname[-16:], '') + 'con_matrix.png'))
#                     fig = plt.figure()
#                     a1 = fig.add_subplot(111)
#                     im = a1.imshow(con_mat, cmap='viridis')
#                     for (j,i),label in np.ndenumerate(con_mat):
#                         if j == 0 and i == 0:
#                             new_label = str(label)+ '\n(hits)'
#                         elif j == 0 and i == 1:
#                             new_label = str(label)+ '\n(misses)'
#                         elif j == 1 and i == 0:
#                             new_label = str(label)+ '\n(false positives)'
#                         else:
#                             new_label = str(label)+ '\n(true negatives)'
# 
#                         a1.text(i,j,new_label,ha='center',va='center', fontname = 'DejaVu Sans',
#                                 fontsize = 12)
#                         
#                     fig.colorbar(im)    
#                     plt.title(f"Confusion matrix of {subID} for {etask}_{eses}", fontname = 'DejaVu Sans')
#                     #plt.savefig(plt_path) #432x288
#                     plt.show()
#                     
#                     check = input("Do you want to proceed? \nyes[y] / no[n]:\n")
#                     if check == 'y':
#                         pass
#                     else:
#                         sys.exit("Script has been interrupted.")
#                     
# =============================================================================
                    # only take matches (= diagonal of confusion matrix [= hits and true negatives])
                    # for each subject, split obj and sc
                    match[isub][itask] = np.sum(match_ret == 1)/(len(match_ret)-miss_cue) + np.sum(match_ret == 4)/(len(match_ret)-miss_cue)

                else:
                    #print(f"{subID}_{etask}_{eses} memory file does not exist.")
                    pass

#sys.exit('stopped before average across sub plot')

# average across all subjects, calculate mean and standard error (axis=0 [vertically across rows])
match = np.ma.array(match, mask=np.isnan(match))
avg_match = np.mean(match,0)
#calculate standard error of the mean, ignoring nans
std_err_match = np.zeros(len(obj_sc))
for i in range(len(obj_sc)):
    SD = np.std(match[:,i], ddof=1)
    std_err = SD/np.sqrt(np.count_nonzero(~np.isnan(match[:,i])))
    std_err_match[i] = std_err
std_err_match = np.ma.array(std_err_match, mask=np.isnan(std_err_match))
    
#std_match = np.nanstd(match,0)

x = np.empty(match.T.shape)
for i in range(match.shape[1]):
    if i == 0:
        x[i] = np.zeros(len(match))
    else:
        x[i] = np.ones(len(match))
color_lst = ['goldenrod', 'royalblue', 'seagreen'] #'seagreen', 'royalblue'
color = np.empty(match.T.shape, dtype = object)
for idx, clr in enumerate(color_lst):
    color[:, idx] = clr

x_title = list()
for i in obj_sc:
    if i == 'obj':
        x_title.append('objects')
    else:
        x_title.append('scenes')

plt.bar(x_title, avg_match, yerr = std_err_match.data, color=['lightgrey','lightblue'])#b.data, std_err_match
        #, fontname = 'DejaVu Sans')
plt.rc('axes', prop_cycle=cycler('color', color_lst))
plt.plot(x_title, match.T)

for i in range(len(obj_sc)):
    plt.scatter(x[i], match.T[i], c=color[i]) 
plt.title(f"Average match across all subjects for {eses}")
plt.ylim(0.7,1)
# =============================================================================
# plt.savefig(os.path.join(save_dir, f"Average match across all subjects for {eses}.png"),
#             dpi=130, bbox_inches='tight')#691x525
# =============================================================================


    
    
    
        