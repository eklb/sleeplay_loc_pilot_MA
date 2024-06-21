#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:48:15 2023

@author: kolbe
"""
#%% LOAD AND DEFINE PATHS...
import os
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from matplotlib import pyplot as plt
from matplotlib import colors 
from matplotlib.lines import Line2D

#from matplotlib import ticker
import pickle
import itertools
#import copy

def extract_cval(clist, extract_val):
    if len(clist) < extract_val:
        raise ValueError(f"Input list has fewer than {extract_val} values.")
    
    step = len(clist) / (extract_val - 1)
    extracted_val = [clist[int(i * step)] for i in range(extract_val - 1)] + [clist[-1]]
    # extracted_val = [clist[int(i * step)] for i in range(extract_val)]
    return extracted_val

# ========================================================================
### DEFINE IMPORTANT PATHS, VARIABLES AND LOAD DECODING FILES
# ========================================================================
# Data from Tardis cluster or local device?
on_tardis = True #False #-- if True, mount Tardis first!

# Plotting wholebrain or within mask smoothed decoding accuracy?
smoothing=True #True -> set 'wholebrain' variable, False = no smoothing
wholebrain=True #False = within mask smoothing

#Cue_type legend: which cue_type (where classifier was trained/tested on) should be plotted?
cue_type=3 # 0=enc, 1=ret, 2=enc2ret, 3=enc&ret
if cue_type==0:
    cue='Enc'
elif cue_type==1:
    cue='Ret'
elif cue_type==2:
    cue='Enc2Ret'
else:
    cue='Enc&Ret'
    
save_name = f"All_cond_ROI_dec.png" #f"Mask_comparison_{cue}_trls_{smooth_type}_sub{sub}.png"
plt_title = f"{cue}"# "Trained/tested on Encoding+Retrieval data"# "Trained on Encoding/Tested on  data"#f"{cue} trial data ({smooth_type} smoothing) - sub{sub}"

# Define subject ID
all_sub = ['01', '02', '03'] #'02'
all_cond = ['Enc', 'Ret', 'Enc2Ret']#, 'Enc&Ret']
which_plot = 'obj' # 'sc'

if on_tardis:
    base_dir = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'TARDIS', 'sleeplay_tardis')
    path_project = os.path.join(base_dir, 'b_loc_v2', 'data')
    
    #/Users/kolbe/Documents/TARDIS/sleeplay_tardis/b_loc_v2/data/decoding/files/sub-01
else:
    base_dir = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'MPIB_NeuroCode', 
                                'Sleeplay', 'data_analysis')
    path_project = os.path.join(base_dir, 'MRI_loc_data')

save_path = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'MPIB_NeuroCode', 
                            'Sleeplay', 'data_analysis', 'MRI_loc_data', 'decoding', 'plots')

os.makedirs(save_path, exist_ok=True)
      
path_decod_files = os.path.join(path_project, 'decoding', 'files')
#path_output_plots = os.path.join(path_project, 'decoding', 'plots')
cmap_path = os.path.join('/Users/kolbe/Documents/my_GitHub/cmap', 'cividisHexValues.txt')
#%%
# load decoding output file from sub specific folder
sub_list = list()
p_list = list()
for isub, sub in enumerate(all_sub):
    sub_path = os.path.join(path_decod_files, f"sub-{sub}", 'observed')
    file_path = os.path.join(sub_path, 'ROI_decod_acc_loc.pkl')
    file_path2 = os.path.join(sub_path, 'INS_control_decod.pkl')
    p_data_dir = os.path.join(save_path, f"ROI_p_vals_sub-{sub}.pkl")
    with open(file_path, "rb") as file:
        decod_data = pickle.load(file)
    with open(file_path2, "rb") as file:
        decod_data2 = pickle.load(file)
    decod_acc = np.concatenate((decod_data['acc'], decod_data2['acc']), axis=0)
    decod_data['acc'] = decod_acc
    decod_data['acc_1dim_label'] = decod_data['acc_1dim_label'] + decod_data2['acc_1dim_label']
    sub_list.append(decod_data)
    # load data with p_vals
    with open(p_data_dir, "rb") as file:
        p_data = pickle.load(file)
    p_list.append(p_data)

# decoding accuries for all masks of obj_ses1 (for enc cue_type only)
acc_obj, acc_sc = list(), list()
p_obj, p_sc = list(), list()
for obs, perm in zip(sub_list, p_list):
    for ises, ses in enumerate(obs['acc_2dim_label']):
        ses_tmp,p_tmp = list(), list()
        for icond, cond in enumerate(all_cond):
            ses_tmp.append(obs['acc'][:,ises,icond]*100)
            p_tmp.append(perm['p_val'][:, ises,icond])
        if ses == 'ses-01' or ses == 'ses-02':
            acc_obj.append(ses_tmp)
            p_obj.append(p_tmp)
        else:
            acc_sc.append(ses_tmp)
            p_sc.append(p_tmp)

if which_plot == 'obj':
    plot_dat = acc_obj
else:
    plot_dat = acc_sc

all2plot = [[[sublist[cond][mask]
            for sublist in plot_dat]
            for cond in range(len(all_cond))]
            for mask in range(len(decod_data['acc_1dim_label']))]            

# acc_mean = list()
# for isess, sess_dat in acc_obj:
#     for cond in len(sess_dat):
        
mean4cond = [[np.mean(mask[cond]) 
             for mask in all2plot]
             for cond in range(len(all_cond))]
        
        
    
acc_obj_mean = np.sum(np.stack(acc_obj), axis=0) / len(acc_obj)
acc_sc_mean = np.sum(np.stack(acc_sc), axis=0) / len(acc_sc)


# # decoding accuries for all masks of obj_ses1 (for enc cue_type only)
# acc_obj_ses2 = decod_data['acc'][:,1,cue_type]
# # decoding accuries for all masks of sc_ses1 (for enc cue_type only)
# acc_sc_ses2 = decod_data['acc'][:,3,cue_type]

#reorganise data into dataframes for easier plotting
# data_allses = {'Objects': acc_obj_mean,
#          'Scenes': acc_sc_mean}
data_allses = {'Encoding (Enc)': mean4cond[0],
         'Retrieval (Ret)': mean4cond[1],
         'Enc2Ret': mean4cond[2]}

# data_ses2 = {'Objects': acc_obj_ses2*100,
#           'Scenes': acc_sc_ses2*100}
# df_allses = pd.DataFrame(data_allses, columns=['Objects', 'Scenes'], index = decod_data['acc_1dim_label'])
df_allses = pd.DataFrame(data_allses, columns=['Encoding (Enc)', 'Retrieval (Ret)', 'Enc2Ret'], index = decod_data['acc_1dim_label'])
# df_ses2 = pd.DataFrame(data_ses2, columns=['Objects', 'Scenes'], index = decod_data['acc_1dim_label'])
mask_labels = decod_data['acc_1dim_label']
#ROI masks to include in plot
mask_labels = decod_data['acc_1dim_label']
incl_masks = ['VIS', 'MTL', 'HPC', 'ERC', 'PPC', 'PREC', 'IPL', 'MOT'] # ['VIS', 'MTL', 'HPC', 'ERC', 'PPC', 'PREC', 'IPL', 'MOT']


df_masks = df_allses.loc[incl_masks]
# df_masks = df_allses.loc[decod_data['acc_1dim_label']]


# Extract indices where values match
match_idx = list()
for idx, mask in enumerate(mask_labels):
    for imask, mask_label in enumerate(incl_masks):
        if mask == mask_label:
            match_idx.append(idx)
            # incl_masks.pop(imask)
match_idx_perm = list()
for idx, mask in enumerate(p_list[0]['p_val_1dim_label']):
    for imask, mask_label in enumerate(incl_masks):
        if mask == mask_label:
            match_idx_perm.append(idx)
     
            
#%% Implement t-test

# from scipy import stats   
# p_vals_masks = np.empty((len(mask_labels), len(all_cond)))
# t_vals_masks = np.empty((len(mask_labels), len(all_cond)))
# sign_lvl_masks = np.empty((len(mask_labels), len(all_cond)))

# sign_levels = [0.05, 0.01, 0.001]
# asterisks = ['1', '2', '3']

# h0 = 10

# for cond in range(len(all_cond)):
#     for i, e in enumerate(all2plot):
#         t_stat, p_val = stats.ttest_1samp(e[cond], h0)
#         p_vals_masks[i][cond] = p_val 
#         t_vals_masks[i][cond] = t_stat 
       

# p_val_df = pd.DataFrame(p_vals_masks, columns = ['Enc', 'Ret', 'Enc2Ret'], index = mask_labels)
# t_stat_df = pd.DataFrame(t_vals_masks, columns = ['Enc', 'Ret', 'Enc2Ret'], index = mask_labels)

# for i,e in np.ndenumerate(p_vals_masks):
#     for level, ast in zip(sign_levels, asterisks):
#         if e < level:
#             sign_lvl_masks[i] = ast
#         # else:
#         #     sign_lvl_masks[i] = 0
            
# s_lvl_df = pd.DataFrame(sign_lvl_masks, columns = ['Enc', 'Ret', 'Enc2Ret'], index = mask_labels)



#%% Plot plot plot...
# =============================================================================
### START PLOTTING    
# =============================================================================
save_name = 'PREC_IPL_decod_acc.png'
fig, ax = plt.subplots()
bar_colors = ['lightsteelblue', 'midnightblue', 'lightgrey']
# legend_colors = ['r','g','b']#['lightgrey', 'lightsteelblue', 'slategrey']
# ses2_colors = ['slategrey']
x_label = 'Applied masks'
y_label = 'Decoding accuracy [%]'

SE_cond = np.zeros((len(all_cond), len(incl_masks)))
SD_cond = np.zeros((len(all_cond), len(incl_masks)))



for cond in range(len(all_cond)):
    for imask, mask in enumerate(match_idx):
        SE_mask = list()
        for dat in plot_dat:
            SE_mask.append(dat[cond][mask])
            
        SE_cond[cond][imask] = np.std(SE_mask, ddof=1) / np.sqrt(len(SE_mask))
        SD_cond[cond][imask] = np.std(SE_mask, ddof=1)
        
bars = sns.barplot(data=df_masks.melt(ignore_index=False).reset_index(),
            x='index', y='value', hue='variable', #ci=("sd"),
            palette=bar_colors,  ax=ax, edgecolor='black', alpha=1, zorder=1)
# ax = sns.barplot(data=df_ses2.melt(ignore_index=False).reset_index(),
#                    x='index', y='value', hue='variable', 
#                    palette=bar_colors2, alpha=0.5) 
# ax = sns.pointplot(data=df_ses2.melt(ignore_index=False).reset_index(),
#                     x='index', y='value', hue='variable', 
#                     palette=ses2_colors, dodge=dogde_width, linestyles='', markers='_') 


# Open the text file for reading
with open(cmap_path, 'r') as file:
    # Read each line from the file and split values by commas
    hex_cmap = [line.strip().split(',') for line in file]  
    
# Flatten the list to convert it into a single list
hex_cmap = [e for i in hex_cmap for e in i]

# color_obj = color_sc[:-2] + color_sc[-1:]
color_obj = extract_cval(hex_cmap, len(acc_obj))
color_sc = extract_cval(hex_cmap, len(acc_sc))

# Example usage:
if which_plot == 'obj':
    scat_col = color_obj
    p_vals = p_obj
else:
    scat_col = color_sc
    p_vals = p_sc

thres_p = 0.05
plot_pos = [-0.27, 0, 0.27] # bar plot position for conditions

for cond in range(len(all_cond)):
    for idx, (obs_acc, p_val) in enumerate(zip(plot_dat, p_vals)):
        shape = ['X' if p_val[cond][p_idx] <= thres_p else '.' for p_idx in match_idx_perm]
        x_coords = np.arange(0,len(incl_masks))+plot_pos[cond]
        x_jitt = x_coords + 0.06 * np.random.rand(len(x_coords))-0.07

        y_coords = obs_acc[cond][match_idx]
        for scat in range(len(x_coords)):
            plt.scatter(x_jitt[scat], y_coords[scat], color=scat_col[idx], marker=shape[scat], edgecolors='dimgrey', linewidths=0.5, zorder=3)#, linestyle='-') #c=np.full(len(e), 0), cmap='cividis'


plt.ylim(0,60)
plt.ylabel(y_label)
plt.xlabel(x_label)
plt.tick_params(axis='x', which='both', bottom=False, top=False)
# plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='dotted', zorder=-1)
# Set the zorder for the bars to a higher value

x_list = []
for bar in bars.patches:
    bar.set_zorder(2)
    x_list.append(bar.get_xy()[0]+(bar._width/2))
x_list = [round(num, 2) for num in x_list]


plt.errorbar(x_list, df_masks.T.to_numpy().flatten(), yerr=SE_cond.T.flatten('F'),
              fmt='-', linestyle='None', color='k', capsize=None, zorder=2)

              # marker='_', mec='blue', zorder=10, elinewidth=1, capsize=2, ecolor='blue',
              # linestyle="None", markersize=10)


plt.axhline(y=10, linewidth=1.2, linestyle='dashed', color='dimgrey', zorder=2)
plt.legend(title='Trained/tested on')
# plt.title(plt_title)
ax.spines[['right', 'top']].set_visible(False)
# Create custom legend handles for marker shapes
shape_legend_handles = [
    Line2D([0], [0], marker='X', color='dimgrey', markerfacecolor='None', linestyle='None', markersize=7, label=f"$p \leq {thres_p}$"),
    # Line2D([0], [0], marker='P', color='w', markerfacecolor='black', markersize=10, label='0.05 < p <= 0.1'),
    Line2D([0], [0], marker='.', color='dimgrey', markerfacecolor='None', linestyle='None', markersize=7, label=f"$p > {thres_p}$")
]

# Get existing handles and labels from the barplot
handles, labels = ax.get_legend_handles_labels()

# Combine barplot handles with custom shape handles
combined_handles = handles + shape_legend_handles
combined_labels = labels + [handle.get_label() for handle in shape_legend_handles]

# Update legend to include all handles and labels
plt.legend(combined_handles, combined_labels)
# plt.legend(handles=combined_handles, title='Trained/tested on', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)


# handles, labels = ax.get_legend_handles_labels()
# handles = [handles[2], handles[3], handles[0], handles[1]] #hacky but works...
#handles.reverse()
# labels=l_labels
#labels.reverse()
# leg_num = len(bar_colors)
# ax.legend(handles[:-1], labels[:-1])  # [-leg_num+1:]use only four last elements for the legend
# plt.savefig(os.path.join(save_path, save_name), dpi=130, bbox_inches='tight')#691x525


#%% Plot ROI masks  
   
