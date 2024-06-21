"""
sleeplay. berlin_loc_v2
prepare data -- (1) save memory data together (incl encoding, retrieval & test outside of scanner) &
(2) score responses of test outside of scanner 
"""

import os
import copy
import numpy as np
import sys
# import pandas as pd
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
import pickle

###### -------- settings
script_dir = os.path.join('C:\\', 'Users', 'elsak', 'OneDrive', 'Dokumente', 'GitHub_c3', 
                          'b_loc_v2')
base_dir = os.path.join('C:\\', 'Users', 'elsak', 'OneDrive', 'Dokumente', 'MPIB_NeuroCode', 
                        'Sleeplay', 'data_analysis') 
#'/Volumes/MPRG-Neurocode/Data/sleeplay_2022_marit/b_loc_v2'
sub_dir = os.path.join(base_dir, 'test_data', 'pilots') #where is the data

skip_catch = bool(0) # if True rating of catch trials is not needed 
save_dir = os.path.join(base_dir, 'test_data', 'analyses') # where data will be saved 
#'behav/00_prep_data_mem' 

stim_dir = os.path.join(script_dir, 'exp_scripts', 'stimuli','images')
sc_dir = 'scenes'
obj_dir = 'obj_all' # folder with all object images without categories as folder structure

sub = ['s04']
ses = ['ses01', 'ses02']
obj_sc = ['obj', 'sc']

isub = 0
obj_sc = 'obj'
#ses = ['ses01']

if obj_sc == 'obj':
    stim_dir = os.path.join(stim_dir, 'obj_all')
    img_ext = '.png'
elif obj_sc == 'sc':
    stim_dir = os.path.join(stim_dir, 'scenes')
    img_ext = '.jpg'

# loop through both session folders to score each retrieval_dict and append to main
for idx, ses_file in enumerate(ses):
    data_dir = os.path.join(sub_dir, sub[isub], 'behav', obj_sc, ses_file)
    
    # load test data
    for i in os.listdir(data_dir):
        if i.startswith('retrieval'):
            ret_fname = i
        elif i.startswith('main_dict'):
            main_file = i 
            
    try: ret_fname
    except NameError: ret_fname = 'not_exists'
    
    # check whether file exists  
    if os.path.exists(os.path.join(data_dir, ret_fname)):
        with open(os.path.join(data_dir, ret_fname), "rb") as f:
            ret_dict = pickle.load(f)
    
        if skip_catch == False:
            ##### rate catch trials & include in memory data
            # create empty column with length of target input
            ret_dict['corr_inp'] = np.zeros(len(ret_dict['targ_inp']), dtype = int)
    
            for i in range(len(ret_dict['targ_inp'])):
                # get response of participant
                print(f"{i+1} of {len(ret_dict['targ_inp'])}")
                resp = ret_dict['targ_inp'][i]
                
                # if they don't know the answer (or running out of time = resp_ is empty == skip)
                if resp != '':
                    # get target image of test + show
                    targ_img = ret_dict[obj_sc + '_ret_targ'][i]
    
                    img = mpimg.imread(os.path.join(stim_dir, targ_img+img_ext))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.show()
                    
                    # rate response of participant & save rating 
                    scoring = True
                    while scoring:
                        print(f"Does '{resp} ' describe the image above?")
                        check = input("yes[y] / no[n]?:\n")
                        if check == 'y':
                            ret_dict['corr_inp'][i] = bool("0")
                            scoring = False
                        elif check == 'n':
                            ret_dict['corr_inp'][i] = bool(0)
                            scoring = False
                        else:
                            pass
        
        main_dict_path = os.path.join(data_dir, main_file)
            
        # check whether main file exists  
        if os.path.exists(main_dict_path):
            with open(main_dict_path, "rb") as f:
                main_file = pickle.load(f)
                main_dict = copy.deepcopy(main_file)
                #main_dict.update(ret_dict)
                main_dict['ret_dict'] = ret_dict
        
        ### save data of test again
        save_name = ret_dict['sub_ID'] + '_' + obj_sc + '_' + ses_file + '_' +'memory_data' + '.pkl'
    
        save_path = os.path.join(save_dir, save_name)
        with open(save_path, 'wb') as f:
            pickle.dump(main_dict, f)
    
    elif not os.path.exists(os.path.join(data_dir, ret_fname)):
        if ses_file == 'ses01':
            print(f"No '{obj_sc}' retrieval_dict for {sub[isub]} in {ses_file} directory. Use different encoding type or sub_ID.")
            pass
        else:
            sys.exit(f"No '{obj_sc}' retrieval_dict for {sub[isub]} in {ses_file} directory. Use different encoding type or sub_ID.")
