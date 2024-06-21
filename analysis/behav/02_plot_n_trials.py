# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:16:36 2023

@author: elsak
"""

import os
import numpy as np
import sys
from matplotlib import pyplot as plt
import pickle
#import copy

# =============================================================================
# SUB_INFO:
# s01: obj ses01, ses02 (+ret_dict); sc ses01 (+ret_dict), ses02 (+ret_dict)
# s02: obj ses01 (+ret_dict), ses02 (+ret_dict); sc ses01 (+ret_dict), ses02 (+ret_dict)
# s03: obj ses01 (half main_dict), ses02; sc ses01 (+ret_dict), ses02
# s04 (=s03): obj ses01 (half main_dict + ret_dict) 
# =============================================================================

script_dir = os.path.join(os.sep, 'Users', 'kolbe', 'Documents','GitHub', 
                          'sleeplay_fMRI_b')
#/Users/kolbe/Documents/GitHub/sleeplay_fMRI_b/stimuli/images/objects
base_dir = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'MPIB_NeuroCode', 
                        'Sleeplay', 'data_analysis', 'behav') 

sub_dir = os.path.join(base_dir, 'behav_loc_data', 'pilots') #where is the data

save_dir = os.path.join(base_dir, 'behav_loc_data', 'analyses') # where data will be saved 

stim_dir = os.path.join(script_dir, 'stimuli','images')
sc_dir = 'scenes'
obj_dir = 'objects' # folder with all object images without categories as folder structure

sub = ['s01','s02', 's03'] #'s01', 's02', 's03', 's04'
ses = ['ses02'] #'ses02'
obj_sc = ['sc'] #'sc'

all_cat = os.listdir(os.path.join(stim_dir, 'objects'))
if '.DS_Store' in all_cat:
    all_cat.remove('.DS_Store')
all_sc = os.listdir(os.path.join(stim_dir, sc_dir))
for i in range(len(all_sc)):
    all_sc[i] = all_sc[i][:-4]

all_enc = np.empty([len(sub), len(all_cat)]) # sub x 2 (= enc/ret_trls)
all_enc[:] = np.nan
all_ret = np.empty([len(sub), len(all_cat)])
all_ret[:] = np.nan
                    
#loop through subjects, enc type and sessions
for isub, subID in enumerate(sub):
    
    for itask, etask in enumerate(obj_sc):
         
        # getting encoding/retrieval data
        for ises, eses in enumerate(ses):
            data_dir = os.path.join(sub_dir, subID, 'behav', etask, eses)
            #print(data_dir)
            
            # load memory data if existent (scored ret_dict appended), otherwise load main dict
            if os.listdir(data_dir) == []:
                sys.exit(f"{subID} has no main_dict for {etask} in {eses}. Exclude sub from list.")
                break
            
            for ifile in os.listdir(data_dir):
                if ifile.startswith('main_dict'):
                    main_dict = ifile
                    with open(os.path.join(data_dir, main_dict), "rb") as file:
                        main_dict = pickle.load(file)
            
            #list with all encoding trials (=24 per categorie)
            enc_trls = list()
            for i in range(len(all_cat)):
                enc_trls.append(main_dict['enc_'+ etask].shape[1]//2)
            
            #save idices in retrieval_rt array where button not pressed (np.nan) = misses
            mri_misses = list()
            n_none = 0
            for idx, trl in np.ndenumerate(main_dict[etask +'_ret_rt'].T):
                if trl is None:
                    n_none += 1
                    pass
                elif np.isnan(trl):
                    mri_misses.append(idx)
                    
            #count misses per obj categorie or sc image and save to dict
            if etask == 'obj':
                targ_list = all_cat
                enc_dict = 'obj_ret_cat'
            else:
                targ_list = all_sc
                enc_dict = 'enc_sc'
                
            count_misses = dict()
            for i in range(len(targ_list)):
                count_misses[targ_list[i]]=0
            for i in mri_misses:
                count_misses[main_dict[enc_dict].T[i]] += 1
            
            #count n retrieval_trials per obj categorie or sc image
            ret_trls = list()
            for i in targ_list:
                ret_trls.append(main_dict['enc_'+ etask].shape[1]//2-count_misses[i])
            
            all_enc[isub] = enc_trls
            all_ret[isub] = ret_trls
            
            
# mean of all encoding and retrieval trials across all subjects for bar plot            
avg_enc = np.mean(all_enc, 0)           
avg_ret = np.mean(all_ret, 0)  
         
# n trials as sum of all encoding and retrieval trials for each subject
n_trls = np.add(all_enc, all_ret)

# calculate standard error of the mean
std_err = np.std(n_trls,0,ddof=1)/np.sqrt(n_trls.shape[1])

# start plotting
fig, ax = plt.subplots()

#x axis indices
x_idc = np.empty(all_enc.shape)
for i in range(len(sub)):
    x_idc[i] = np.arange(len(targ_list))

#stacked bar plot with encoding and retrieval trials
plt.bar(targ_list, avg_enc, color='lightgrey')
bars = plt.bar(targ_list, avg_ret, yerr=std_err, bottom=avg_enc, color='lightsteelblue')

# =============================================================================
# #plot number of enc/ret trials in plot
# for idx, bar in enumerate(bars):
#     y_val = bar.get_height() + avg_enc[idx]
#     plt.text(bar.get_x(), y_val + 4, f"n={int(y_val)}", fontsize=9.5)
#     plt.text(bar.get_x()+0.2, avg_ret[idx]+18, int(avg_ret[idx]), fontsize=9.5, c='steelblue')
#     plt.text(bar.get_x()+0.2, avg_enc[idx]-3, int(avg_enc[idx]), fontsize=9.5, c='grey')
# =============================================================================

# =============================================================================
# #specify colour list dependent on length of sub list
# sub_colors = ['goldenrod', 'royalblue', 'seagreen', 'navy']
# color_lst = list()
# for i in range(len(sub)):
#     color_lst.append(sub_colors[i])
#     
# #colour matrix for scatter plot
# color = np.empty(all_enc.T.shape, dtype = object)
# for idx, clr in enumerate(color_lst):
#     color[:, idx] = clr
# 
# #plot inidividual data points with line
# for idx, val in np.ndenumerate(all_enc):
#     plt.scatter(x_idc[idx], all_enc[idx]+all_ret[idx], color=color.T[idx])
# for i,e in enumerate(all_ret):
#     plt.plot(x_idc[i], n_trls[i], marker='.', linestyle='-', color=color_lst[i]) 
# 
# plt.grid(axis='y', linestyle='dotted')
# =============================================================================
plt.xticks(rotation=45)
#plt.yticks(list(plt.yticks()[0]) + x_tick)
plt.yticks(np.arange(0, 60, 5))
for index, label in enumerate(ax.get_yticklabels()):
    if index % 2 != 0:
        label.set_visible(False)
plt.ylabel('n trials')

c_legend = {'retrieval trials':'lightsteelblue', 'encoding trials':'lightgrey'}         
labels = list(c_legend.keys())
handles = list()
for label in labels:
    handles.append(plt.Rectangle((0,0),1,1, color=c_legend[label]))
plt.legend(handles, labels)

if obj_sc[0] == 'obj':
    title = f"N trials of all subjects across object categories ({eses})"
else:
    title = f"N trials of all subjects across scenes ({eses})"
plt.title(title)
# =============================================================================
# plt.savefig(os.path.join(save_dir, f"{obj_sc[0]}_decoding_trials_{eses}_std_err.png"),
#         dpi=130, bbox_inches='tight')#691x525
# =============================================================================
   
   
