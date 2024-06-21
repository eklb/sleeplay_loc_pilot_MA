
# This script restructures behavioural data to be more easily used for sleeplay_decoding
# output = pd with columns defining:
#   (1) run no
#   (2) trial_phase (fixation cross, encoding cue, retrieval cue)
#   (3) category (will be used as class labels during decoding)
#   (4) time (time point in sec, will be translated into TRs before decoding)
#   (5) reponses (whether key was pressed during retrieval)
#   

import os
import pandas as pd
import numpy as np
import copy
#import pickle5 as pickle


print(f"pandas version: {pd.__version__}")
path_root = os.getcwd()
if path_root[:12]=='/Users/kolbe':
    
    run_clust = False 
    path_project = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'MPIB_NeuroCode', 
                                'Sleeplay', 'data_analysis', 'behav')
    path_behav = os.path.join(path_project, 'behav_loc_data', 'pilots') # behavioural data (for time points of TRs)
    save_dir = os.path.join(path_project, 'decod_prep_data') # path to save prepped data
    
else:
    
    run_clust = True
    path_project = os.path.join(os.sep,'home', 'mpib', 'kolbe', 'sleeplay_tardis', 'behav')
    
    save_dir = os.path.join(path_project, 'decod_prep_data') # path to save prepped data on cluster


sub = [ '02'] # ['01', '02'] # '03', '04'
ses = ['1', '2'] # ['1', '2']
enc_type = ['obj'] # 'sc', 'obj'

nblocks_in_run = 4 # how many blocks/run
ntrls_in_block = 5 # encoding and retrieval trials are taken together
nvars = 5 # n of trial_phases that should be saved
ntrls_run = nblocks_in_run*ntrls_in_block

enc_cue_dur = 4
ret_cue_dur = 5


# enc_type = 'obj' # required to load behaviour
# if ses == '3' or ses == '4': # scene sessions
#     enc_type = 'sc'
    

for itype in enc_type:
    if itype == 'obj':
        # define dict with all stim names per category
        cat_base = {'animal': ['giraffe', 'panda', 'cat', 'kangaroo', 'sheep', 'dog'],
                    'clothing': ['cardigan', 'jacket', 'jeans', 'shorts', 'skirt', 'vest'],
               'kitchen': ['kettle', 'grater', 'iron', 'mixer', 'rollingpin', 'toaster'],
               'body': ['calf', 'ear', 'foot', 'hand', 'knee', 'shoulder'],
               'fruit': ['apple', 'banana', 'blueberries', 'cherry', 'melon', 'strawberry'],
               'furniture': ['bed', 'bench', 'coffeetable', 'couch', 'stool', 'table'],
               'music': ['clarinet', 'guitar', 'piano', 'saxophone', 'violine', 'xylophone'],
               'sports': ['baseball', 'basket', 'basketball', 'boxing', 'golf', 'trampoline'],
               'tool': ['hammer', 'pliers', 'saw', 'scraper', 'screwdriver', 'wrench'],
               'vehicle': ['airplane', 'bike', 'camper', 'car', 'motorcycle', 'snowmobile']
               }
        # dict: get the key if string is in value -> for object category extraction 
        def matchingKeys(dict_, string_):
            return [key for key,val in dict_.items() if any(string_ in s for s in val)]
        
    for isub in sub:
        
        for ises in ses:
            if run_clust:
                
                main_dict_path = os.path.join(path_project, 'behav_loc_data')
                path_file = os.path.join(main_dict_path, 'main_dict_s' + isub + '_loc' + ises + '_' + itype + '.pickle')
                
                # # as pandas version does not support protocol 5 
                # with open(path_file, "rb") as fh:
                #     data = pickle.load(fh)    
                
            else:
                
                main_dict_path = os.path.join(path_behav, f"s{isub}", 'behav', itype, f"ses0{ises}")
                path_file = os.path.join(main_dict_path, 'main_dict_s' + isub + '_loc' + ises + '_' + itype + '.pickle')

            data = pd.read_pickle(path_file)
    
            nblocks = np.shape(data[f"t_{itype}_enc_fix"])[1]
            nruns = int(nblocks/nblocks_in_run)
            runs_idx = np.arange(0, nblocks, nblocks_in_run) # block indices when a new run starts
            
            if itype == 'obj':
                enc_fix_idx = np.arange(0, ntrls_run*nvars, nvars) #
                enc_cue_idx = np.arange(1, ntrls_run*nvars, nvars) #
                ret_fix_idx = np.arange(2, ntrls_run*nvars, nvars) #
                ret_cue_idx = np.arange(4, ntrls_run*nvars, nvars) #
                ret_resp_idx = np.arange(5, ntrls_run*nvars, nvars) #
            
                #runs_idx = np.arange(0, ntrls_run*nvars, nblocks_in_run) # 
        
            
            fix_del = copy.deepcopy(data[f"t_{itype}_enc_fix"][0,0]) # this is the delay time between scanner trigger and onset of first fix cross 
            #del_odd = 13 # how long does the odd even lasts
            
            for irun, erun in enumerate(runs_idx):
                
                # get indices for run start and run end in block matrix
                run_start = erun
                run_end = run_start + nblocks_in_run        
                
                ### DELAYS. get all the time delays relevant to calculate the run start time (this sucks...)
                enc_iti = data[f"ITI_{itype}_enc"][:,run_start:run_end]
                #enc_iti = enc_iti.flatten('F')
                ret_iti = data[f"ITI_{itype}_ret"][:,run_start:run_end]
                #ret_iti = ret_iti.flatten('F')
                
                ################################################
                ###### CATEGORIES
                ################################################
                
                ## get ENCODING (have to be rated first since just single objects are saved)
                if itype == 'obj':
                    # get encoded objects for each run separately
                    enc_obj = copy.deepcopy(data['enc_obj'][:,run_start:run_end])
                    enc_obj = enc_obj.flatten('F')
                    enc_cat_tmp = np.empty(ntrls_run, dtype = 'object')
                    
                    # extract all categories from trial data of each run (ntrls_run)
                    for itrl in range(ntrls_run):
                        enc_obj_tmp = enc_obj[itrl]
                        enc_cat_tmp[itrl] = matchingKeys(cat_base, enc_obj_tmp)[0]
                    
                    ## get RETRIEVAL
                    ret_cat_tmp = data['obj_ret_cat'][:,run_start:run_end]
                    ret_cat_tmp = ret_cat_tmp.flatten('F')
                    
                else:
                    enc_cat_tmp = copy.deepcopy(data['enc_sc'][:,run_start:run_end])
                    enc_cat_tmp = enc_cat_tmp.flatten('F')
                    
                    ## get RETRIEVAL
                    ret_cat_tmp = copy.deepcopy(data['sc_ret_targ'][:,run_start:run_end])
                    ret_cat_tmp = ret_cat_tmp.flatten('F')
                
               
                
                ################################################
                ###### EXTRACT IMPORTANT TIME STAMPS
                ################################################
                
                ########## ENCODING - fixcross
                # extract all encoding fixcross onsets from data of each run
                enc_fix_tmp = copy.deepcopy(data[f"t_{itype}_enc_fix"][:,run_start:run_end])
                
                if irun !=0: # in first run -> no need to change anything in time stamps
                    # set scanner start at t0, account for scanner delay (fix_del)
                    enc_fix_tmp = enc_fix_tmp - (enc_fix_tmp[0][0] - fix_del)
                
                    #enc_fix_tmp = enc_fix_tmp.flatten('F')
            
                    ########## ENCODING - object cue 
                    enc_cue_tmp = enc_fix_tmp + enc_iti # just add the fix cross ITIs to it
            
                    
                    ########## RETRIEVAL - fixcross
                    # get delay of odd even task (how long did it last)
                    enc_cue = copy.deepcopy(data[f"t_{itype}_enc"][:,run_start:run_end])
                    ret_fix = copy.deepcopy(data[f"t_{itype}_ret_fix"][:,run_start:run_end])
                    blocks=ret_fix.shape[1]
                    odd_even_dur = np.empty(blocks)
                    ret_fix_tmp = np.empty(ret_fix.shape)
                    ret_fix1 = np.empty(blocks)
                    
                    # 4 blocks per run
                    for i in range(blocks):
                        # from last enc cue end to first retrieval fix cross onset
                        odd_even_dur[i] = ret_fix[0,i] - (enc_cue[4,i] + enc_cue_dur)
                        
                        del_ = enc_cue_tmp[4,i] + enc_cue_dur + odd_even_dur[i] # delay from last obj trial of block + odd even
                        
                        #calculate time stamp of each first ret fix cross per block (relative to run start)
                        ret_fix1[i] = enc_fix_tmp[0,0] + del_
                        
                        #extract absolute values for ret fix crosses per block
                        ret_fix_bl = copy.deepcopy(ret_fix[:,i])
                        if itype == 'obj':
                            ret_fix_tmp[:,i] = ret_fix_bl - (ret_fix[0][i] - ret_fix1[i])
                        else:
                            ret_fix_tmp[:,i] = (ret_fix_bl - ret_fix[0][i]) + ret_fix1[i]
            
                    ########## RETRIEVAL. cue
                    ret_cue_tmp = ret_fix_tmp + ret_iti
            
                    ########## RETRIEVAL. response
                    ret_resp = copy.deepcopy(data[f"{itype}_ret_rt"][:,run_start:run_end])
                    ret_resp_tmp = ret_cue_tmp + ret_resp
                    
                else: # first run. just copy time stamps from output file
                    enc_cue_tmp = copy.deepcopy(data[f"t_{itype}_enc"][:,run_start:run_end])
                    ret_fix_tmp = copy.deepcopy(data[f"t_{itype}_ret_fix"][:,run_start:run_end])
                    ret_cue_tmp = copy.deepcopy(data[f"t_{itype}_ret_cue"][:,run_start:run_end])
                    ret_resp_tmp = ret_cue_tmp + copy.deepcopy(data[f"{itype}_ret_rt"][:,run_start:run_end])
                        
                
                
                ########## bring everything in the same shape 
                enc_fix_tmp = enc_fix_tmp.flatten('F')
                enc_cue_tmp = enc_cue_tmp.flatten('F')
                ret_fix_tmp = ret_fix_tmp.flatten('F')
                ret_cue_tmp = ret_cue_tmp.flatten('F')
                ret_resp_tmp = ret_resp_tmp.flatten('F')
                ret_resp_tmp = np.asarray(ret_resp_tmp, dtype = float)
                
                
                
                enc_fix_label = np.repeat('enc_fix', len(enc_fix_tmp))
                enc_cue_label = np.repeat('enc_cue', len(enc_cue_tmp))
                ret_fix_label = np.repeat('ret_fix', len(ret_fix_tmp))
                ret_cue_label = np.repeat('ret_cue', len(ret_cue_tmp))
                ret_resp_label = np.repeat('ret_resp', len(ret_resp_tmp))
                
                # labeling responses during retrieval, include 0 if no response
                resp_tmp = np.ones(ntrls_run, dtype = int)
                resp_tmp2 = np.ones(ntrls_run, dtype = int)
                for itrl in range(ntrls_run-1):
                    if np.isnan(ret_resp_tmp[itrl]) == True: 
                        resp_tmp2[itrl] = 0
                
                # bring everything in series shape (per run):
                #trial type labels
                trial_phase_tmp = np.hstack((enc_fix_label, enc_cue_label, ret_fix_label, ret_cue_label, ret_resp_label))
                # relativ time stamps for all trial types per run
                time_tmp = np.hstack((enc_fix_tmp, enc_cue_tmp, ret_fix_tmp, ret_cue_tmp, ret_resp_tmp))
                # corresponding categories
                category_tmp = np.hstack((enc_cat_tmp, enc_cat_tmp, ret_cat_tmp, ret_cat_tmp, ret_cat_tmp))
                # whether trials are usable or not (import for retrieval trials on button presses)
                response_tmp= np.hstack((resp_tmp, resp_tmp, resp_tmp, resp_tmp2, resp_tmp2))
                
                
                ################################################
                ###### put everything together in run vectors
                ################################################
                # time_tmp = np.empty(ntrls_run*nvars, dtype = float)
                # trial_phase_tmp = np.empty(ntrls_run*nvars, dtype = object)
                # category_tmp = np.empty(ntrls_run*nvars, dtype = object)
                # response_tmp = np.ones(ntrls_run*nvars, dtype = int)
                
                # for itrl in range(ntrls_run-1):
                    
                #     # encoding_fixation 
                #     time_tmp[enc_fix_idx[itrl]] = enc_fix_tmp[itrl]
                #     trial_phase_tmp[enc_fix_idx[itrl]] = 'enc_fix'
                #     category_tmp[enc_fix_idx[itrl]] = enc_cat_tmp[itrl]
                #     #response_tmp[enc_fix_idx[itrl]] = 1
                
                #     # encoding_cue
                #     time_tmp[enc_cue_idx[itrl]] = enc_cue_tmp[itrl]
                #     trial_phase_tmp[enc_cue_idx[itrl]] = 'enc_cue'
                #     category_tmp[enc_cue_idx[itrl]] = enc_cat_tmp[itrl]
                #     #response_tmp[enc_cue_idx[itrl]] = 1
                    
                #     # retrieval_fixation 
                #     time_tmp[ret_fix_idx[itrl]] = ret_fix_tmp[itrl]
                #     trial_phase_tmp[ret_fix_idx[itrl]] = 'ret_fix'
                #     category_tmp[ret_fix_idx[itrl]] = ret_cat_tmp[itrl]
                #     #response_tmp[ret_fix_idx[itrl]] = 1
                    
                #     # retrieval_cue
                #     time_tmp[ret_cue_idx[itrl]] = ret_cue_tmp[itrl]
                #     trial_phase_tmp[ret_cue_idx[itrl]] = 'ret_cue'
                #     category_tmp[ret_cue_idx[itrl]] = ret_cat_tmp[itrl]
                #     if np.isnan(ret_resp_tmp[itrl]) == True: 
                #         response_tmp[ret_cue_idx[itrl]] = 0
                    
                #     # retrieval_response
                #     time_tmp[ret_resp_idx[itrl]] = ret_resp_tmp[itrl]
                #     trial_phase_tmp[ret_resp_idx[itrl]] = 'ret_resp'
                #     category_tmp[ret_resp_idx[itrl]] = ret_cat_tmp[itrl]
                #     if np.isnan(ret_resp_tmp[itrl]) == True: 
                #         response_tmp[ret_resp_idx[itrl]] = 0
                    
                    
                # create array with run number (corresponding to data shape from above)
                run_no_tmp = np.repeat(irun, ntrls_run*nvars)
                
                # append all data for all runs 
                if irun == 0:  
                    run_no = copy.deepcopy(run_no_tmp)
                    category = copy.deepcopy(category_tmp)
                    trial_phase = copy.deepcopy(trial_phase_tmp)
                    time = copy.deepcopy(time_tmp)
                    response = copy.deepcopy(response_tmp)
                    
                else:
                    run_no = np.hstack((run_no, run_no_tmp))
                    
                    category = np.hstack((category, category_tmp))
                    trial_phase = np.hstack((trial_phase, trial_phase_tmp))
                    time = np.hstack((time, time_tmp))
                    response = np.hstack((response, response_tmp))
                    
                    
            run = run_no +1
            
        
            dat_out = {'run': pd.Series(run),
                       'category': category,
                       'trial_phase': trial_phase,
                       'time': time,
                       'response': response}
            
            dat_out = pd.Series(data = dat_out)
            # adapt session naming for obj and sc 
            if itype == 'obj':
                ses_nr = ises
            else:
                ses_nr = str(int(ises) + 2)
                
            path_save = os.path.join(save_dir, 'behav_sub-' + isub + '_ses-0' + ses_nr + '.pkl')
            dat_out.to_pickle(path_save)
            
            
        
            
            