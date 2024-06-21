# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:16:36 2023

@author: elsak
"""

#%% Importand define...
import os
import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib import colors 
from matplotlib.patches import Patch
import seaborn as sns
import ptitprince as pt
import sys
import pickle
import pandas as pd
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
all_cat = os.listdir(os.path.join(stim_dir, 'objects'))
if '.DS_Store' in all_cat:
    all_cat.remove('.DS_Store')
    
cmap_path = os.path.join('/Users/kolbe/Documents/my_GitHub/cmap', 'cividisHexValues.txt')


loc_behav_dir = os.path.join(base_dir, 'behav_loc_all', 'main_dicts')
mem_dat_dir = os.path.join(base_dir, 'behav_loc_all', 'scored_dicts')

all_sub = ['s01','s02', 's03', 's04'] #'s01', 's02', 's03', 's04'
ses = ['ses01'] #'ses02'
obj_sc = ['obj', 'sc'] #'sc'
press_cond = ['obj', 'sc', 'all']

plot_mem = True

def extract_cval(clist, extract_val):
    if len(clist) < extract_val:
        raise ValueError(f"Input list has fewer than {extract_val} values.")
    
    step = len(clist) / (extract_val - 1)
    extracted_val = [clist[int(i * step)] for i in range(extract_val - 1)] + [clist[-1]]
    # extracted_val = [clist[int(i * step)] for i in range(extract_val)]
    return extracted_val

#%% Load all data
behav_list = [i for i in os.listdir(loc_behav_dir)]
mem_list = [i for i in os.listdir(mem_dat_dir)]

for i in behav_list:
    if i=='.DS_Store' or i=='._.DS_Store':
        behav_list.remove(i)
behav_list.sort()

all_dat = [pd.read_pickle(os.path.join(loc_behav_dir, i)) for i in behav_list]
all_mem = [pd.read_pickle(os.path.join(mem_dat_dir, i)) for i in mem_list]

for idx, mem_file in enumerate(all_mem):
    if mem_file['sub_ID'] == 's04': # remove subject 4 mem file from list due to 
        del all_mem[idx]


# split data for obj and scene localiser
obj_dat, sc_dat = list(), list()
obj_idx, sc_idx = list(), list()
for idx, main_dict in enumerate(all_dat):
    enc_type = main_dict['loc_sess'][0][:-21]
    if enc_type == 'obj':
        obj_dat.append(main_dict)
        obj_idx.append(idx)
    else:
        sc_dat.append(main_dict)
        sc_idx.append(idx)

idx_obj, idx_sc = list(), list()
obj_mem, sc_mem = list(), list()
for idx, mem_file in enumerate(all_mem):
    enc_type = mem_file['loc_sess'][0][:-21]
    if enc_type == 'obj':
        obj_mem.append(mem_file)
        idx_obj.append(idx)
    else:
        sc_mem.append(mem_file)
        idx_sc.append(idx)
        
#%% EXTRACT OVERALL BUTTON PRESSES AND HITS

press_rate_all = np.empty(len(all_dat), dtype=object)
press_rate_obj =np.empty(len(obj_dat), dtype=object)
press_rate_sc =np.empty(len(sc_dat), dtype=object)

press_cond = [all_dat, obj_dat, sc_dat]
press_rates = [press_rate_all, press_rate_obj, press_rate_sc]

for icond, cond in enumerate(press_cond):
    for idx, main_dict in enumerate(cond): 
        enc_type = main_dict['loc_sess'][0][:-21]
    
        all_rets = main_dict[enc_type +'_ret_rt'] # Extract dict including all retrieval trial RTs
        
        mask_nones = ~np.array([trl is None for idx, trl in np.ndenumerate(all_rets.T)]) # create boolean matrix where all NoneType objects are False
        # mask_nones = ~np.isnan(np.array(ret_press, dtype=float))
        ret_trls = all_rets.flatten('F')[mask_nones] # extract all valid retrieval trials, excluding Nones
        # ret_misses = np.isnan(np.array(ret_trls, dtype=float)) # extract all nans (aka misses)  
        # count_misses = np.count_nonzero(ret_misses)
        ret_press = ~np.isnan(np.array(ret_trls, dtype=float)) # extract and count all button presses
        count_press = np.count_nonzero(ret_press)
        press_perc = count_press/len(ret_press)*100
        
        press_rates[icond][idx] = press_perc

# calculate all means
press_means = np.zeros(len(press_cond))
press_SE = np.zeros(len(press_cond))
for idx, rate in enumerate(press_rates):
    press_means[idx] = np.mean(rate)
    press_SE[idx] = np.std(rate, ddof=1) / np.sqrt(len(rate))
    
#%% RETRIEVAL COMPARISON

ret_comp = [all_mem, obj_mem, sc_mem]

ret_match_all = np.empty(len(all_mem), dtype=object)
ret_match_obj = np.empty(len(obj_mem), dtype=object)
ret_match_sc = np.empty(len(sc_mem), dtype=object)

ret_match_rates = [ret_match_all, ret_match_obj, ret_match_sc]

for iret, mem_ret in enumerate(ret_comp):
    for imem, mem_file in enumerate(mem_ret):
        enc_type = mem_file['loc_sess'][0][:-21]
        # match_ret = np.zeros([len(mem_file['ret_dict']['corr_inp'])])
        ret_match = 0
        cue_loss = 0
        
        for itrl, icue in enumerate(mem_file['ret_dict'][f"{enc_type}_ret_cues"]):
            cue_idx = np.array(np.where(mem_file[f"{enc_type}_ret_cues"] == icue))
            if cue_idx.size == 0:
                cue_loss += 1
            else:
                cue_idx = cue_idx[:, cue_idx[1,:].argsort()]
                # saving only idx of 2nd cue repetition 
                cue_idx = (cue_idx[0,-1], cue_idx[1][-1])
                # get reaction time of last retrieval cue
                rt_cue = mem_file[f"{enc_type}_ret_rt"][cue_idx]
                # get response during test
                corr_inp = mem_file['ret_dict']['corr_inp'][itrl]
                
                if ~np.isnan(rt_cue) and corr_inp == 1:
                    ret_match += 1
                
        match_perc = ret_match/(len(mem_file['ret_dict']['corr_inp'])-cue_loss)*100
        
        ret_match_rates[iret][imem] = match_perc
        
ret_match_means = np.zeros(len(ret_match_rates))
ret_match_SE = np.zeros(len(ret_match_rates))
for idx, rate in enumerate(ret_match_rates):
    ret_match_means[idx] = np.mean(rate)
    ret_match_SE[idx] = np.std(rate, ddof=1) / np.sqrt(len(rate))


#%% PLOT PLOT PLOT

if plot_mem:
    y_label = 'Average Retrieval Match [%] \n(scan- vs. post-scan)'
    var2plot = ret_match_means #ret_match_means #press_means
    sub_all = ret_match_all #ret_match_all #press_rate_all
    sub_obj = ret_match_obj #ret_match_obj #press_rate_obj
    sub_sc = ret_match_sc
    
    sub_idx_obj = idx_obj
    sub_idx_sc = idx_sc
    SE = ret_match_SE
    save_name = 'Avg_ret_match.png'
else:
    y_label = 'Average Retrieval Button Presses [%]'
    var2plot = press_means
    sub_all = press_rate_all 
    sub_obj = press_rate_obj 
    sub_sc = press_rate_sc
    
    sub_idx_obj = obj_idx
    sub_idx_sc = sc_idx
    SE = press_SE
    save_name = 'Avg_ret_presses.png'

#%%BOXPLOT for objects only
plot_name = 'Avg_ret_presses_obj_boxplt.png'
plot_save_dir = os.path.join(base_dir, 'behav_loc_data', 'analyses') # where data will be saved 

with open(cmap_path, 'r') as file:
    # Read each line from the file and split values by commas
    hex_cmap = [line.strip().split(',') for line in file]  
    
# Flatten the list to convert it into a single list
hex_cmap = [e for i in hex_cmap for e in i]

sub_colors = np.array(extract_cval(hex_cmap, len(sub_obj)))
sub_colors = np.array(extract_cval(hex_cmap, len(sub_sc)))

# sub_col_obj = sub_colors[sub_idx_obj]
# sub_col_sc = sub_colors[sub_idx_sc]
#%%
fig, ax = plt.subplots()

# plt.boxplot(sub_obj)
# plt.figure(figsize=(6,6))
sub_sc2 = np.empty(4, dtype=object)
sub_sc2[0]=sub_sc[0]
sub_sc2[1:]=sub_sc[2:]


ret_match_SE = np.std(sub_sc2, ddof=1) / np.sqrt(len(sub_sc2))
ret_press_SE = press_SE[1]

plot_var = sub_obj
plot_err = ret_press_SE

with open(cmap_path, 'r') as file:
    # Read each line from the file and split values by commas
    hex_cmap = [line.strip().split(',') for line in file]  

hex_cmap = [e for i in hex_cmap for e in i]

sub_colors = np.array(extract_cval(hex_cmap, len(plot_var)))

box_col = 'dimgrey'
box_zord = 10
box_alpha = 1
ax=pt.half_violinplot(data = plot_var, color = 'lightgrey', 
                      bw = 0.5, cut = 3, scale = "area", width = 0.6, inner = None,
                      offset=0.15, linewidth=1.6, ax=ax)#offset=outline of plot on x , palette = col

ax=sns.boxplot(data = plot_var, zorder=box_zord,
               #width = 0.12, 
               showfliers=True, 
               boxprops = {'linewidth': 1.2, 'facecolor':'none', 'alpha': box_alpha, 
                           'edgecolor': box_col, 'zorder': box_zord},
               whiskerprops = {'linewidth':1.2, 'color': box_col, 'alpha': box_alpha,
                               'zorder': box_zord}, 
               medianprops = {'linewidth':1.2, 'color': box_col, 'zorder': box_zord},
               capprops = {'linewidth':1.2, 'color': box_col, 'alpha': box_alpha,
                           'zorder': box_zord},
               width=0.2,
               ax=ax)
               # position=)
x_pos_rc=-0.15 #x position in plot
# ax=plt.plot([x_pos_rc, x_pos_rc], [ci_top, ci_bottom], color='black',
#             linewidth=2.3)
ax=plt.plot(x_pos_rc, np.mean(plot_var), color='lightsteelblue', marker='.', 
            markeredgecolor='black', markersize=15, zorder=5)#x=-0.105
ax=plt.errorbar(x_pos_rc, np.mean(plot_var), yerr=plot_err, fmt='-', color='k', capsize=None, elinewidth=1.4)


for isub, sub in enumerate(plot_var):
    plt.scatter(0, sub, color=sub_colors[isub], marker='o', zorder=12)#, linestyle='-') #c=np.full(len(e), 0), cmap='cividis'

# for patch in ax.artists:
#     bbox = patch.get_bbox()
#     bbox.x0 -= 0.5
#     bbox.x1 -= 0.5
#     patch.set_bbox(bbox)

# ax.set_ylabel("Average button presses [%]", fontsize=20)
# ax.set_yticklabels(labels=(range(-2,3,1)), fontsize=18)
# ax.set(xlabel='',
#         xticks=[],
#         xticklabels=[],
#         #ylim=(-3,2.8),
#         yticks=(range(-2,3,1)))
# plt.xticks([])
# plt.xlabel('')
# plt.ylim(60,104)
plt.ylim(60,104)

plt.ylabel("Average button presses [%]", fontsize=15)
# plt.ylabel('Average Retrieval Match [%] \n(scan- vs. post-scan)', fontsize=15)

#ax = plt.gca()
# ax.spines['bottom'].set_position('zero')
# ax.spines[['right', 'top']].set_visible(False)
plt.gca().xaxis.set_visible(False)
plt.gca().tick_params(axis='y', labelsize=14)


plt.grid(axis='y', linestyle='dotted')
plt.savefig(os.path.join(save_dir, plot_name), dpi=130, bbox_inches='tight')#691x525

#%% PLOT BEHAV BARS


fig, ax = plt.subplots()
bar_colors = ['midnightblue', 'slategrey', 'lightsteelblue']
l_labels = ['Combined trials', 'Objects', 'Scenes']#, 'Ses-02','Ses-02']

# Create bar plot
for i, value in enumerate(var2plot):
    plt.bar(i, value, color=bar_colors[i])
    plt.errorbar(i, value, yerr=SE[i], fmt='-', color='k', capsize=None)
    
# plt.bar(range(len(ret_match_means)), ret_match_means, color=bar_colors)

# Open the text file for reading
with open(cmap_path, 'r') as file:
    # Read each line from the file and split values by commas
    hex_cmap = [line.strip().split(',') for line in file]  
    
# Flatten the list to convert it into a single list
hex_cmap = [e for i in hex_cmap for e in i]

sub_colors = np.array(extract_cval(hex_cmap, len(sub_all)))
sub_col_obj = sub_colors[sub_idx_obj]
sub_col_sc = sub_colors[sub_idx_sc]
# for cond in range(len(all_cond)):
for isub, sub in enumerate(sub_all):
    plt.scatter(0, sub, color=sub_colors[isub], marker='.')#, linestyle='-') #c=np.full(len(e), 0), cmap='cividis'
for isub, sub in enumerate(sub_obj):
    plt.scatter(1, sub, color=sub_col_obj[isub], marker='.')#, linestyle='-') #c=np.full(len(e), 0), cmap='cividis'
for isub, sub in enumerate(sub_sc):
    plt.scatter(2, sub, color=sub_col_sc[isub], marker='.')#, linestyle='-') #c=np.full(len(e), 0), cmap='cividis'


# Add custom legend box
legend_handles = [Patch(facecolor=color, label=label) for color, label in zip(bar_colors, l_labels)]
plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)


# plt.bar(np.arange(0,3), ret_match_means, color=bar_colors)
plt.ylim(0,100)
plt.yticks(range(0, 101, 20))
plt.ylabel(y_label)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.grid(axis='y', linestyle='dotted', zorder=0)
plt.gca().set_axisbelow(True)
ax.spines[['right', 'top']].set_visible(False)

# plt.savefig(os.path.join(save_dir, save_name), dpi=130, bbox_inches='tight')#691x525



#%% PLOT MISSES PER OBJECT CATEGORY
ret_dict = dict()
for dat_idx,dat in enumerate(all_dat):
    if dat['loc_sess'][0][:3] == 'obj':
        print(dat['sub_ID'])
        #save idices in retrieval_rt array where button not pressed (np.nan) = misses
        mri_misses = list()
        n_none = 0
        for idx, trl in np.ndenumerate(dat['obj_ret_rt'].T):
            if trl is None:
                n_none += 1
                pass
            elif np.isnan(trl):
                mri_misses.append(idx)
                
        count_misses = dict()
        for i in range(len(all_cat)):
            count_misses[all_cat[i]]=0
        for i in mri_misses:
            count_misses[dat['obj_ret_cat'].T[i]] += 1
        
        #count n retrieval_trials per obj categorie or sc image
        ret_trls = list()
        for i in all_cat:
            ret_trls.append(dat['enc_obj'].shape[1]//2-count_misses[i])
        print(ret_trls)
        
        ret_dict[dat['sub_ID'] + f"_{dat_idx}"] = ret_trls

# average ret trials per class 
ret_class = list()
for key in ret_dict.keys():
    ret_dict[key] = np.array(ret_dict[key])
    ret_class.append(np.mean(ret_dict[key]))
    # print(np.mean(ret_dict[key])) 
    
#combine s03+04 since same subject
ret_sub3 = sum(ret_class[-2:])/len(ret_class[-2:])
ret_class = ret_class[:5]
ret_class.append(ret_sub3)

# calculate SE
ret_SE = np.std(np.array(ret_class), ddof=1 / np.sqrt(len(ret_class)))

n_enc_trls = 24
enc_class = [n_enc_trls]*len(ret_class)

trls2plot = [np.array(enc_class), np.array(ret_class)]
trls_SE = [0, ret_SE]

#plot bar plots

fig, ax = plt.subplots()
plt.figure(figsize=(4,3))

trl_plot_name = 'Avg_trls_per_class'
bar_colors = ['lightsteelblue', 'midnightblue']#, 'lightsteelblue']
x_labels = ['Encoding', 'Retrieval']#, 'Ses-02','Ses-02']

# Create bar plot
for i, trls in enumerate(trls2plot):
    plt.bar(i, np.mean(trls), color=bar_colors[i], edgecolor='black', zorder=2)
    plt.errorbar(i, np.mean(trls), yerr=trls_SE[i], fmt='-', color='k', capsize=None)



    
# plt.bar(range(len(ret_match_means)), ret_match_means, color=bar_colors)

# Open the text file for reading
with open(cmap_path, 'r') as file:
    # Read each line from the file and split values by commas
    hex_cmap = [line.strip().split(',') for line in file]  
    
# Flatten the list to convert it into a single list
hex_cmap = [e for i in hex_cmap for e in i]
sub_col = np.array(extract_cval(hex_cmap, len(ret_class)))

#plot subject scatter plots
#with jitter 
x_scat = 1 # position of scatter plot axis on x  
x_jitt = x_scat + 0.09 * np.random.rand(len(ret_class))-0.05
for isub, sub in enumerate(ret_class):
    plt.scatter(x_jitt[isub], sub, color=sub_col[isub], marker='o', edgecolors='dimgrey', zorder=3)#, linestyle='-') #c=np.full(len(e), 0), cmap='cividis'


plt.ylim(0,26)
plt.yticks(np.arange(0, 26, 6), fontsize=12)

# plt.ylabel("Average trials per class", fontsize=15)
plt.xticks([0,1], x_labels, fontsize=12)
#ax = plt.gca()
# ax.spines['bottom'].set_position('zero')
ax.spines[['right', 'top']].set_visible(False)
# plt.gca().xaxis.set_visible(False)
# plt.gca().tick_params(axis='y', labelsize=14)
plt.grid(axis='y', linestyle='dotted')

# plt.savefig(os.path.join(save_dir, trl_plot_name), dpi=130, bbox_inches='tight')#691x525

    
    

# plt.savefig(os.path.join(save_dir, f"{obj_sc[0]}_decoding_trials_{eses}_std_err.png"),
#         dpi=130, bbox_inches='tight')#691x525
# =============================================================================
   
   
