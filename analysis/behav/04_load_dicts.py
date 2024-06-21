#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:26:56 2024 

CREATE LIST OF ALL BEHAV DICTS (from b_loc_v2)

@author: kolbe
"""

import os
import pandas as pd

which_dicts = 'mem' # 'ret', 'mem'

if which_dicts == 'main':
    path_file = '/Users/kolbe/Documents/MPIB_NeuroCode/Sleeplay/data_analysis/behav/behav_loc_all/main_dicts'
elif which_dicts == 'ret':
    path_file = '/Users/kolbe/Documents/MPIB_NeuroCode/Sleeplay/data_analysis/behav/behav_loc_all/retrieval_dicts'
else:
    path_file = '/Users/kolbe/Documents/MPIB_NeuroCode/Sleeplay/data_analysis/behav/behav_loc_all/scored_dicts'

behav_list = [i for i in os.listdir(path_file)]

for i in behav_list:
    if i=='.DS_Store' or i=='._.DS_Store':
        behav_list.remove(i)
behav_list.sort()

all_data = [pd.read_pickle(os.path.join(path_file, i)) for i in behav_list]

#%% CALCULATE RUN LENGTHS BASED ON LOG FILE

##### calculate run length based on log file time stamps
behav_OI = all_data[5]

enc_type = 'sc' # 'obj'

#define run number according to corresponding block number
nruns = 10 # 12 for obj, 10 for sc
run_nr = 8 

run_dict = {
    1:0,
    2:4,
    3:8,
    4:12,
    5:16,
    6:20,
    7:24,
    8:28,
    9:32,
    10:36,
    11:40,
    12:44
    }

for i in range(1, nruns+1):
    
    ### start (first fix_cross onset)
    irun_st = run_dict[i]#run_nr
    irun_end = irun_st + 3
    
    first_fix = behav_OI[f't_{enc_type}_enc_fix'][0,irun_st]
    last_cue = behav_OI[f't_{enc_type}_ret_cue'][4,irun_end] 
    
    run_end = last_cue + 5 + 6.5 # cue onset + cue duration + last_fix
    
    run_total = run_end - first_fix # in s
    
    print(f"Run {i}: Run length = {run_total} s")
    




