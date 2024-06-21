# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:39:13 2022

@author: kolbe
"""


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
from instructions_loc_ver2 import *


# =============================================================================
# Define all directories
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


# =============================================================================
# Define dicts for screen times and keys
# =============================================================================
win_time = dict()
win_time["run_fix"] = 6.5 #6.5
win_time["pract_fix"] = 4 #4
win_time["pract_stim"] = 4 #4
win_time["stim_time"] = 1 #4
win_time["ret_wait"] = 1 #5
#win_time["enc_break"] = 10 #10
win_time["fix_break"] = 5 #5
win_time["exit_time"] = 2 #2
win_time["task_text"] = 2 #screen time for intro/outro for odd/even task 

key_dict = dict()
key_dict["sub_key"] = "b"
key_dict["pract_key"] = "space"
key_dict["pract_even"] = "right"
key_dict["pract_odd"] = "left"
key_dict["scanner_key"] = "t"
key_dict["operator_key"] = "space"
key_dict["exit_key"] = "escape"
key_dict["even"] = "z"
key_dict["odd"] = "b"

# =============================================================================
# Define all functions for experiment
# =============================================================================

def fill_matrix(matrix_name, values, shuffle_idx1, shuffle_idx2):
    """
    Function to create either object or object path matrix from empty matrix.
    Creating temporary list for each block, containing all objects/object paths 
    of a block (20 trials) which is then inserted into columns of matrix. 
    (1 column = 1 block)
    matrix_name = obj_matrix1/obj_paths1
    values = all_obj/path_list
    """
    
    temp_list = []
    for i in range(matrix_name.shape[1]):
        temp_list.append(list())
        
    temp_values = copy.deepcopy(values)
    
    for i in range(len(temp_values)):
        for j in range(len(temp_list)//2):
            pick = random.choice(temp_values[i])
            temp_values[i].remove(pick)
            temp_list[j].append(pick)
                 
    for i in range(len(temp_list)):
        random.seed(shuffle_idx1[i])
        random.shuffle(temp_list[i])
        
    for i in range(len(temp_list)//2,len(temp_list)):
        temp_list[i] = copy.deepcopy(temp_list[i-6])
        random.seed(shuffle_idx2)
        random.shuffle(temp_list[i])
    
    for i in range(len(temp_list)):
        matrix_name[:,i] = temp_list[i]
        

def fill_cues(cue_list, matrix):
# Read in verbs into obj_verbs
    temp_list = []
    temp_cues = copy.deepcopy(cue_list)
    
    for i in range(matrix.shape[1]):
        temp_list.append(temp_cues[:len(matrix[:,i])])
        del temp_cues[:len(matrix[:,i])]
        matrix[:, i] = temp_list[i]
    

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

def new_matrix(old_mat, new_mat): 
    """
    Function to create a new stimuli, paths and word matrix for shorter 
    encoding blocks.
    
    Passed arguments:
        old_mat = obj_matrix1, obj_paths1, obj_verbs1, ITI_
        new_mat = obj_matrix, obj_paths, obj_verbs
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
   
def call_text(win, text):
    """
    Function to call, draw and flip a text stimulus given the text input.
    """
    text_stim = visual.TextStim(win, text=text, font = "Open Sans", wrapWidth=1400,
                                height=52, color=(-1,-1,-1))
    text_stim.draw()
    win.flip()
    
def fix_cross(win, symbol):
    """
    Function to show a fixation cross or question mark  given the wanted 
    string input as argument. 
    """
    fix = visual.TextStim(win, text=symbol, height=46, color=(-1,-1,-1))
    fix.draw()
    win.flip()

def exit_exp(win):
    """
    Function to exit experiment by key pressing "q".
    """
    kb = keyboard.Keyboard()       
    keys = kb.getKeys(["q"])
    if "q" in keys: # press q to exit experiment
        # save main dict if earlier exited
        with open(file_path, "wb") as file:
            pickle.dump(main_dict, file, pickle.HIGHEST_PROTOCOL)   
            
        call_text(win, exit_text)
        kb.waitKeys(keyList=["escape"])
        win.close()
        core.quit()

def enc_pract(img_stim, img_path, word_cue, win, fix_time, stim_time):
    """
    Practice session of encoding task. Will start automatically if sub_ID of
    dialogue box is "sXX".
    
    Passed arguments: img_stim = pract_obj/pract_sc, 
    img_path = obj_paths, 
    word_cue = obj_verbs, 
    win = p_win, 
    fix_time/stim_time = wintime dict 
    """

    for i in range(len(img_stim)):
        exit_pract(win)
        print(f"Practice Encoding : Trial {i+1}")
                    
        fix_cross(win, fix)
        core.wait(fix_time)
        
        img_targ = visual.ImageStim(win, image=os.path.join(img_path, 
                                      img_stim[i]), pos=(0,-win.size[1]*0.05), 
                                      size=(400,400))
        enc_cue = visual.TextStim(win, text=word_cue[i], pos=(0,win.size[1]*0.125), 
                                  color=(-1,-1,-1), height=52)
        
        img_targ.draw()
        enc_cue.draw()
        win.flip()
        core.wait(stim_time)
        
def ret_pract(pract_cues, win, fix_time, stim_time, ret_wait):
    """
    Practice session of retrieval task.
    
    Passed arguments:
    pract_cues = pract_verbs, pract_adj, 
    win = p_win,
    fix_time/stim_time = wintime dict
    """
    
    random.shuffle(pract_cues)     
    
    for i in range(len(pract_cues)): 
        exit_pract(win)
        
        print(f"Practice Retrieval: Trial {i+1}")
        fix_cross(win, fix)
        core.wait(fix_time)
        
        
        word_cue = visual.TextStim(win, text=pract_cues[i], color=(-1,-1,-1), 
                                   pos=(0,win.size[1]*0.125), height=52) 
        q_mark = visual.TextStim(win, text="?", color=(-1,-1,-1), height=52)
        word_cue.draw()
        q_mark.draw()
        kb.clock.reset()
        win.flip()
        
        key_press = kb.waitKeys(maxWait=ret_wait, keyList=["space"], waitRelease=True)
        if key_press is None:
            pass
        else:
            word_cue = visual.TextStim(win, text=pract_cues[i], color=(-1,-1,-1), 
                                       pos=(0,win.size[1]*0.125), height=52) 
            q_mark = visual.TextStim(win, text="?", color=(-0.529411764705882, 
                                                           0.403921568627451, 
                                                           -0.113725490196078), #mediumseagreen
                                     height=52)
            word_cue.draw()
            q_mark.draw()
            win.flip()
            core.wait(ret_wait-key_press[0].rt)
            
def exit_pract(win):
    """
    Function to exit experiment by key pressing "q".
    """
    kb = keyboard.Keyboard()       
    keys = kb.getKeys(["q"])
    if "q" in keys: # press q to exit experiment
        call_text(win, exit_text)
        kb.waitKeys(keyList=["escape"])
        win.close()
        core.quit()        
            
def enc_block(img_stim, img_path, word_cue, t_fix, t_enc, block, win, fix_ITI, 
              stim_time, ext):
    """
    Structure of encoding task of one block. Presenting a verb cue on top of 
    an image target. Image stimuli can be either objects or scenes depending 
    on the encoding type (chosen in dialogue box). 
    
    Passed arguments: img_stim = obj_matrix, img_path = obj_paths, 
    word_cue = obj_verbs/sc_adj, t_fix = t_obj/sc_enc_fix, 
    t_enc = t_obj/sc_enc, 
    block = block (number based on dialogue box input),
    win = win, 
    fix_ITI = ITI_obj/sc_enc (np array for jitter),
    stim_time = wintime dict,
    ext = obj_ext/sc_ext (filetype extension)
    """

    for i in range(len(img_stim[:,block])):
        exit_exp(win)
        print(f"Block_{block + 1} : Trial {i+1}")
                    
        fix_cross(win, fix)
        t_fix[i,block] = clock.getTime()
        core.wait(fix_ITI[i,block])
        
        img_targ = visual.ImageStim(win, image=os.path.join(img_path[i,block], 
                                      img_stim[i,block]+ext), pos=(0,-win.size[1]*0.05)) #size=(400,400)
        enc_cue = visual.TextStim(win, text=word_cue[i,block], pos=(0,win.size[1]*0.125), 
                                  color=(-1,-1,-1), height=52)
        
        img_targ.draw()
        enc_cue.draw()
        win.flip()
        t_enc[i,block] = clock.getTime()
        core.wait(stim_time)
                        
            
def ret_block(img_stim, cue, enc_cat_mat, ret_cat_mat, t_fix, t_cue, cue_array, 
              targ_array, rt_array, block, win, fix_ITI, stim_time, ret_wait, dict_key):
    """
    Random retrieval of object or scene images of preceding encoding block by 
    presenting previously paired verb cue alone. After each stimulus, subjects
    are shown a question mark instead of a fixation cross and have to press
    spacebar wihtin 5 s when image is remembered. Reaction times are saved in 
    ms. If not image is not remembered, next cue is presented after 5 s.
    Cues and targets are additionally saved for each retrieval block.
    
    Passed arguments:
    img_stim = obj/sc matrix, 
    cue = obj_verbs, sc_adj, 
    enc_cat_matrix = enc cat (as copy for enc cats), sc_paths (as placeholder), 
    cat_matrix = obj_cat (to save ret cats), sc_paths (as placeholder),
    t_fix/cue = t_obj/sc_ret_fix/cue,
    cue/targ/rt_array = empty sc/obj_cues/targ/rt,
    block = block,
    win = win,
    fix_ITI = ITI_obj/sc_ret (np array for jitter)
    """
    
    #Create cues for later retrieval and targets for main dict within each block
    ret_cues = copy.deepcopy(cue[:,block])
    ret_targ = copy.deepcopy(img_stim[:,block])
    ret_cat = copy.deepcopy(enc_cat_mat[:,block])
    
    # use same shuffle index 
    seed3 = np.random.randint(0,100)
    
    rdm_state = np.random.RandomState(seed3)
    rdm_state.shuffle(ret_cues)
    rdm_state.seed(seed3)
    rdm_state.shuffle(ret_targ)
    rdm_state.seed(seed3)
    rdm_state.shuffle(ret_cat)
     
    #react_times = list()
    for i in range(len(cue[:,block])): 
        exit_exp(win)
        print(f"Block_{block + 1} : Retrieval Trial {i+1}")
        cue_array[i,block] = ret_cues[i]
        targ_array[i,block] = ret_targ[i]
        ret_cat_mat[i,block] = ret_cat[i]
        
        fix_cross(win, fix)
        t_fix[i,block] = clock.getTime()
        core.wait(fix_ITI[i,block])
        
        word_cue = visual.TextStim(win, text=ret_cues[i], color=(-1,-1,-1), 
                                   pos=(0,win.size[1]*0.125), height=52) 
        q_mark = visual.TextStim(win, text="?", color=(-1,-1,-1), height=52)
        word_cue.draw()
        q_mark.draw()
        kb.clock.reset()
        win.flip()
        
        t_cue[i,block] = clock.getTime()
        key_press = kb.waitKeys(maxWait=ret_wait, keyList=[dict_key], waitRelease=True)
    
        if key_press is None:
            rt_array[i,block] = np.nan
            #react_times.append("No key press")
        else:
            rt_array[i,block] = key_press[0].rt
            #react_times.append(key_press[0].rt)
            word_cue = visual.TextStim(win, text=ret_cues[i], color=(-1,-1,-1), 
                                       pos=(0,win.size[1]*0.125), height=52) 
            q_mark = visual.TextStim(win, text="?", color=(-0.529411764705882, 
                                                           0.403921568627451, 
                                                           -0.113725490196078), #mediumseagreen
                                     height=52)
            word_cue.draw()
            q_mark.draw()
            win.flip()
            core.wait(ret_wait-key_press[0].rt)

    # append cues, targets, rts for each block to numpy arrays   
    #cue_array[:,block] = list(ret_cues)
    #targ_array[:,block] = list(ret_targ)
    #rt_array[:,block] = react_times
    
def color_text(win, text, color):
    """
    Function to call, draw and flip a text stimulus given the text input.
    """
    text_stim = visual.TextStim(win, text=text, font = "Open Sans", wrapWidth=1400,
                                height=52, color=color)
    text_stim.draw()
    win.flip()
    
def odd_even(win, task_len, intro_text, task_text1, task_text2, task_text3, 
             green, red, key_even, key_odd, win_time):
    """
    Function for an odd/even number task as short attention test.
    
    Passed arguments:
        win = psychopy window
        task_len = length of task in s
        intro_text = intro_task
        task_text1
        task_text2
        task_text3
        green = color code for green
        red = color code for red
        key_even = key_dict["even"]
        key_odd = key_dict["odd"]
        win_time = win_time["task_wait"]
       
    """
    kb = keyboard.Keyboard()
    
    call_text(win, intro_text)
    core.wait(win_time)
    
    # define numpy array to draw number from
    nr = np.arange(1,10)
    # start clock
    clock = core.Clock()
    # set variables to zero for later counts
    time, trial, wrongs, rights = 0, 0, 0, 0
    
    while time < task_len:
        time = clock.getTime()
        trial += 1
        rdm_nr = np.random.choice(nr)
        
        nr_stim = visual.TextStim(win, text=rdm_nr, font = "Open Sans", height=100, 
                                  color=(-1,-1,-1), pos=(0,win.size[1]*0.125))
        
        txt_stim = visual.TextStim(win, text=task_text1, font = "Open Sans", height=60, 
                                  wrapWidth=1400, color=(-1,-1,-1), pos=(0,-win.size[1]*0.3))
        nr_stim.draw()
        txt_stim.draw() 
        win.flip() 
        
        key = kb.waitKeys(keyList=([key_even, key_odd]))
    
        if rdm_nr % 2 == 0 and key_even in key:
            rights += 1
            color_text(win, task_text2, green)
            core.wait(0.4)
        elif rdm_nr % 2 != 0 and key_odd in key:   
            rights += 1
            color_text(win, task_text2, green)
            core.wait(0.4)
    
        else:
            wrongs += 1
            color_text(win, task_text3, red)
            core.wait(0.4)
            
    outro_task = f"""
    Performance Score: {rights} von {trial} 
    \n\n\n\n >>> Retrieval Test
    """ 
    call_text(win, outro_task)
    core.wait(win_time)  

           
# =============================================================================
# Create dialogue box for experiment information. All following conditions based
# on input.
# =============================================================================

exp_info = {"sub_ID" : "sXX", "loc_sess" : ["loc1", "loc2"],
            "enc_type" : ["obj_enc", "sc_enc"], 
            "block_no" : "block_1",
            "language" : ["german", "english"]}
exp_gui = gui.DlgFromDict(exp_info, 
                         title="Please fill in the following fields:",
                         labels={"sub_ID" : "Enter Subject_ID \n(e.g. sXX):",
                                 "loc_sess" : "1st or 2nd Localiser Session?",
                                 "enc_type" : "Object or Scene encoding?",
                                 "block_no" : "Starting from which block? \n(Default set to block_1)",
                                 "language" : "Choose language:"},
                         order=["sub_ID", "loc_sess", "enc_type", "block_no", 
                                "language"]) 
                                 
if exp_gui.OK:
    check_gui = gui.Dlg(title="Input summary")
    check_gui.addText("Please check your input:")
    check_gui.addText(f"Subject ID: {exp_info['sub_ID']}")
    check_gui.addText(f"Encoding type: {exp_info['enc_type']}")
    check_gui.addText(f"Starting from: {exp_info['block_no']}")
    check_gui.addText("\nClick 'OK' to Continue and 'Cancel' to Re-enter info\n")
    check_gui.show()

    if check_gui.OK:
            print('Experiment info double-checked')
    else:
        exp_gui.show()
        if exp_gui.OK:
            check_gui.show()
            if check_gui.OK:
                print('Experiment info double-checked')
        else:
            print('Input Cancelled')
            sys.exit("Running of script has been interrupted")
    
else:
    print('Input Cancelled')
    sys.exit("Running of script has been interrupted")
#Save actual date and time in expinfo 
date = datetime.datetime.today()
exp_info["test_date"] = date.strftime("%d/%m/%Y/%H:%M")


#Check whether main dict file exists for sub ID input
if exp_info["enc_type"] == "obj_enc":
    
    file_path = os.path.join(output_dir, f"{exp_info['sub_ID']}", f"main_dict_{exp_info['sub_ID']}_{exp_info['loc_sess']}_obj.pickle")
    file_path2 = os.path.join(output_dir,  'back_up', f"bup_{exp_info['sub_ID']}_{exp_info['loc_sess']}_{exp_info['enc_type']}.pickle")
else:
    file_path = os.path.join(output_dir, f"{exp_info['sub_ID']}", f"main_dict_{exp_info['sub_ID']}_{exp_info['loc_sess']}_sc.pickle")
    file_path2 = os.path.join(output_dir, 'back_up', f"bup_{exp_info['sub_ID']}_{exp_info['loc_sess']}_{exp_info['enc_type']}.pickle")    

path_name, file_name = os.path.split(file_path)
os.makedirs(path_name, exist_ok=True)

path_name, file_name = os.path.split(file_path2)
os.makedirs(path_name, exist_ok=True)

file_exist = os.path.exists(file_path)

# =============================================================================
# 
# warn_gui = gui.Dlg(title="FILE WARNING")
# warn_gui.addText("File with subject ID already exists. Please double-check!")
# warn_gui.addText("\nClick 'OK' to Continue and 'Cancel' to Re-enter info\n")
# if file_exist and exp_info["block_no"] == "block_1":
#     warn_gui.show()
#     if warn_gui.OK:
#         pass
#     else:
#         sys.exit("Running of script has been interrupted")
#     
# =============================================================================
if file_exist and exp_info["block_no"] != "block_1":
    #warn_gui.show()
    with open(file_path, "rb") as file:
        main_dict = pickle.load(file)
        
        # save all required variables from main_dict
        if "enc_obj" in main_dict:
            obj_matrix = main_dict["enc_obj"]
            obj_paths = main_dict["obj_paths"]
            obj_cat = main_dict["obj_ret_cat"]
            obj_verbs = main_dict["enc_obj_verbs"]
            t_obj_enc_fix = main_dict["t_obj_enc_fix"]
            t_obj_ret_fix = main_dict["t_obj_ret_fix"]
            t_obj_enc = main_dict["t_obj_enc"]
            t_obj_ret_cue = main_dict["t_obj_ret_cue"]
            obj_targ = main_dict["obj_ret_targ"]
            obj_cues = main_dict["obj_ret_cues"]
            obj_rt = main_dict["obj_ret_rt"]
            ITI_obj_enc = main_dict["ITI_obj_enc"]
            ITI_obj_ret = main_dict["ITI_obj_ret"]
            
            # take one example image obj (from first categorie folder) to save 
            # file extension name as variable
            obj_name1, obj_ext = os.path.splitext(os.listdir(os.path.join(obj_dir, obj_cat[0,0]))[0])

            
        if "enc_sc" in main_dict:
            sc_matrix = main_dict["enc_sc"]
            sc_paths = main_dict["sc_paths"]
            sc_adj = main_dict["enc_sc_adj"]
            t_sc_enc_fix = main_dict["t_sc_enc_fix"]
            t_sc_ret_fix = main_dict["t_sc_ret_fix"]
            t_sc_enc = main_dict["t_sc_enc"]
            t_sc_ret_cue = main_dict["t_sc_ret_cue"]
            sc_targ = main_dict["sc_ret_targ"]
            sc_cues = main_dict["sc_ret_cues"]
            sc_rt = main_dict["sc_ret_rt"]
            ITI_sc_enc = main_dict["ITI_sc_enc"]
            ITI_sc_ret = main_dict["ITI_sc_ret"]
            
            # take first sc example image to save file extension name as variable
            sc_name1, sc_ext = os.path.splitext(os.listdir(sc_dir)[0])

                   
        
#if file_exist and exp_info['loc_sess'] == 'loc2' or not file_exist:
else:                              
# =============================================================================
# Load data and create matrices dependent on dialogue box input
# =============================================================================

    if exp_info["enc_type"] == "obj_enc" and exp_info["sub_ID"] != "sXX":
            
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
        
        # Define all empty numpy arrays for object encoding
        obj_matrix1 = np.empty([10, 12], dtype = object)
        obj_paths1 = np.empty([10, 12], dtype = object)
        obj_verbs1 = np.empty([10, 12], dtype = object)
        
        obj_matrix = np.empty([5, 48], dtype = object)
        obj_paths = np.empty([5, 48], dtype = object)
        obj_verbs = np.empty([5, 48], dtype = object)
        
        obj_cat = np.empty([5, 48], dtype = object)
        obj_targ = np.empty([5, 48], dtype = object)
        obj_cues = np.empty([5, 48], dtype = object)
        obj_rt = np.empty([5, 48], dtype = object)
        
        # arrays for time stamps
        t_obj_enc_fix = np.empty([5, 48], dtype = object)
        t_obj_enc = np.empty([5, 48], dtype = object)
        t_obj_ret_fix = np.empty([5, 48], dtype = object)
        t_obj_ret_cue = np.empty([5, 48], dtype = object)
        
        # arrays for ITIs
        ITI_obj_enc1 = np.empty([10, 24], dtype = object)
        ITI_obj_ret1 = np.empty([10, 24], dtype = object)
        
        ITI_obj_enc = np.empty([5, 48], dtype = object)
        ITI_obj_ret = np.empty([5, 48], dtype = object)
        
        #create ITIs
        for i in range(obj_matrix1.shape[1]*2):
            tmp = makeITI()
            ITI_obj_enc1[:,i] = tmp[:obj_matrix1.shape[0]]
            ITI_obj_ret1[:,i] = tmp[obj_matrix1.shape[0]:]
            
    
        # set shuffle index to have same for obj and path matrix when calling fill_matrix    
        idx_list = set(list(range(0,100)))
        shuffle_idx1 = random.sample(idx_list, 12)
        shuffle_idx2 = random.randint(0,100)
    
        # Read in obj and obj paths data into empty arrays
        fill_matrix(obj_matrix1, all_obj, shuffle_idx1, shuffle_idx2)
        fill_matrix(obj_paths1, path_list, shuffle_idx1, shuffle_idx2)
    
        # Read in verb stimuli from excel file
        if exp_info["language"] == "german":
            df2 = pd.read_excel(os.path.join(wordcue_dir, 'verbs_german.xlsx'))
        else:
            df2 = pd.read_excel(os.path.join(wordcue_dir, 'verbs_english.xlsx'))
        df2.columns = ["verbs"] # name column of table
        verb_list = df2["verbs"].tolist() # store verbs in list
        random.shuffle(verb_list)
    
        # Create verb matrix of half length
        fill_cues(verb_list, obj_verbs1)
        
        # Deepcopy obj matrices for later concatenation of copy to existing matrix 
        obj_copy = copy.deepcopy(obj_matrix1)
        path_copy = copy.deepcopy(obj_paths1)
        verb_copy = copy.deepcopy(obj_verbs1)
        
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
        
        # Concatenate all matrices to 20x12 shape
        obj_matrix1 = np.concatenate((obj_matrix1, obj_copy), axis=1)
        obj_paths1 = np.concatenate((obj_paths1, path_copy), axis=1)
        obj_verbs1 = np.concatenate((obj_verbs1, verb_copy), axis=1)
        
       
        #Create new matrices(5,48) for shorter encoding
        new_matrix(obj_matrix1, obj_matrix)
        new_matrix(obj_paths1, obj_paths)
        new_matrix(obj_verbs1, obj_verbs)
        
        new_matrix(ITI_obj_enc1, ITI_obj_enc)
        new_matrix(ITI_obj_ret1, ITI_obj_ret)
            
        #Create object category matrix from obj_paths (analogous to encoding)
        enc_cat = copy.deepcopy(obj_paths)
        for i in range(obj_cat.shape[0]):
            for j in range(obj_cat.shape[1]):
                enc_cat[i,j] = os.path.basename(os.path.normpath(enc_cat[i,j]))
     
    elif exp_info["enc_type"] == "sc_enc" and exp_info["sub_ID"] != "sXX":
        
        # Create list of all scene images     
        sc_stim = os.listdir(sc_dir)
        if '.DS_Store' in sc_stim:
            sc_stim.remove('.DS_Store')
        for i in range(len(sc_stim)):
            sc_name, sc_ext = os.path.splitext(sc_stim[i])
            sc_stim[i] = sc_name
        
        
        # Create list of sc paths with len(sc_stim) for image resizing
        sc_dirs = []
        sc_dirs.extend([sc_dir]*len(sc_stim))
        #img_resize(sc_dirs, sc_stim, (700,420)) #size-presentation 508x304,8
        
        
        # Define all empty numpy arrays for sc encoding
        sc_matrix1 = np.empty([10, 10], dtype = object)
        sc_paths1 = np.empty([10, 10], dtype = object)
        sc_adj1 = np.empty([10, 10], dtype = object)
        
        sc_matrix = np.empty([5, 40], dtype = object)
        sc_paths = np.empty([5, 40], dtype = object)
        sc_adj = np.empty([5, 40], dtype = object)
        
        sc_targ = np.empty([5, 40], dtype = object)
        sc_cues = np.empty([5, 40], dtype = object)
        sc_rt = np.empty([5, 40], dtype = object)
        
        t_sc_enc_fix = np.empty([5, 40], dtype = object)
        t_sc_enc = np.empty([5, 40], dtype = object)
        t_sc_ret_fix = np.empty([5, 40], dtype = object)
        t_sc_ret_cue = np.empty([5, 40], dtype = object)
        
        # arrays for ITIs
        ITI_sc_enc1 = np.empty([10, 20], dtype = object)
        ITI_sc_ret1 = np.empty([10, 20], dtype = object)
        
        ITI_sc_enc = np.empty([5, 40], dtype = object)
        ITI_sc_ret = np.empty([5, 40], dtype = object)
        
        #create ITIs
        for i in range(sc_matrix1.shape[1]*2):
            tmp = makeITI()
            ITI_sc_enc1[:,i] = tmp[:sc_matrix1.shape[1]]
            ITI_sc_ret1[:,i] = tmp[sc_matrix1.shape[1]:]
            
        # Read in sc stim and sc paths data into empty arrays
        temp_list = []
        for i in range(sc_matrix1.shape[1]):
            sc_block = copy.deepcopy(sc_stim)
            random.shuffle(sc_block)
            temp_list.append(sc_block)
        
        for i in range(len(temp_list)):
            sc_matrix1[:,i] = temp_list[i]
            sc_paths1[:,i] = sc_dirs
        
        # Read in sc adjectives
        if exp_info["language"] == "german":
            df3 = pd.read_excel(os.path.join(wordcue_dir, 'adjectives_german.xlsx'))
        else:
            df3 = pd.read_excel(os.path.join(wordcue_dir, 'adjectives_english.xlsx'))
        df3.columns = ["adjectives"] 
        adj_list = df3["adjectives"].tolist()
        random.shuffle(adj_list)
        
        #Create adj matrix of half length
        fill_cues(adj_list, sc_adj1)
    
        # Deepcopy sc matrices for later concatenation of copy to existing matrix 
        sc_copy = copy.deepcopy(sc_matrix1)
        adj_copy = copy.deepcopy(sc_adj1)
        
        # Shuffle all belonging matrices with same shuffle index over first dimension, 
        # keeping column structure, set seed to random number for each run of script
        seed2 = np.random.randint(0,100)
    
        rdm_state = np.random.RandomState(seed2)
        rdm_state.shuffle(sc_copy)
        rdm_state.seed(seed2)
        rdm_state.shuffle(adj_copy) 
    
        # Concatenate all matrices to 10x20 shape
        sc_matrix1 = np.concatenate((sc_matrix1, sc_copy), axis=1)
        sc_adj1 = np.concatenate((sc_adj1, adj_copy), axis=1)
        sc_paths1 = np.concatenate((sc_paths1, sc_paths1), axis=1)
        
        #Create new matrices(5,48) for shorter encoding
        new_matrix(sc_matrix1, sc_matrix)
        new_matrix(sc_paths1, sc_paths)
        new_matrix(sc_adj1, sc_adj)
        
        new_matrix(ITI_sc_enc1, ITI_sc_enc)
        new_matrix(ITI_sc_ret1, ITI_sc_ret)
    
if exp_info["sub_ID"] == "sXX":
    pass

else:
    #Create main dict containing all important values for each subject
    loc_date = "_".join([exp_info["enc_type"], exp_info["test_date"]])
    
    # =============================================================================
    # if os.path.exists(file_path) and exp_info['block_no'] != 'block_1':
    #     with open(file_path, "rb") as file:
    #         main_dict = pickle.load(file)
    # =============================================================================
    if exp_info["block_no"] == "block_1":     
        main_dict = {}
        main_dict["sub_ID"] = exp_info["sub_ID"]
        main_dict["loc_sess"] = [loc_date]
        
    else:
        with open(file_path, "rb") as file:
            main_dict = pickle.load(file)
       
        main_dict["loc_sess"].append(loc_date)
            
    if exp_info["enc_type"] == "obj_enc":
        main_dict["enc_obj"] = obj_matrix
        main_dict["obj_paths"] = obj_paths
       # main_dict["obj_img_ext"] = obj_ext
        main_dict["obj_ret_cat"] = obj_cat
        main_dict["enc_obj_verbs"] = obj_verbs
        main_dict["obj_ret_cues"] = obj_cues
        main_dict["obj_ret_targ"] = obj_targ
        main_dict["obj_ret_rt"] = obj_rt
        main_dict["t_obj_enc_fix"] = t_obj_enc_fix
        main_dict["t_obj_enc"] = t_obj_enc
        main_dict["t_obj_ret_fix"] = t_obj_ret_fix
        main_dict["t_obj_ret_cue"] = t_obj_ret_cue
        main_dict["ITI_obj_enc"] = ITI_obj_enc
        main_dict["ITI_obj_ret"] = ITI_obj_ret
    
    else:
        main_dict["enc_sc"] = sc_matrix
        main_dict["sc_paths"] = sc_paths
       # main_dict["sc_img_ext"] = sc_ext
        main_dict["enc_sc_adj"] = sc_adj
        main_dict["sc_ret_cues"] = sc_cues
        main_dict["sc_ret_targ"] = sc_targ
        main_dict["sc_ret_rt"] = sc_rt
        main_dict["t_sc_enc_fix"] = t_sc_enc_fix
        main_dict["t_sc_enc"] = t_sc_enc
        main_dict["t_sc_ret_fix"] = t_sc_ret_fix
        main_dict["t_sc_ret_cue"] = t_sc_ret_cue
        main_dict["ITI_sc_enc"] = ITI_sc_enc
        main_dict["ITI_sc_ret"] = ITI_sc_ret
        
    #sys.exit("Running of script has been interrupted")

# =============================================================================
# Start experiment.
# =============================================================================      

# Practice session
if exp_info["sub_ID"] == "sXX":
    if exp_info["enc_type"] == "obj_enc":
        
        # Create list with all practice object images  
        p_obj_stim = os.listdir(p_obj_dir)
        if '.DS_Store' in p_obj_stim:
            p_obj_stim.remove('.DS_Store')
        
        # Load practice word cues
        df = pd.read_excel(os.path.join(wordcue_dir, 'practice_cues.xlsx'))
        pract_cues = df.iloc[:,0].tolist() # store verbs in list
        
        pract_verbs = pract_cues[:3]
    
    else:
        
        # Create list with all practice scene images  
        p_sc_stim = os.listdir(p_sc_dir)
        if '.DS_Store' in p_sc_stim:
            p_sc_stim.remove('.DS_Store')
        
        # Load practice word cues
        df = pd.read_excel(os.path.join(wordcue_dir, 'practice_cues.xlsx'))
        pract_cues = df.iloc[:,0].tolist() # store verbs in list
        
        pract_adj = pract_cues[3:6]
        
            
    p_win = visual.Window([1440,900], monitor="testMonitor", fullscr=False, 
                        units='pix', 
                        color=light_grey)
    p_win.mouseVisible = False
    
    kb = keyboard.Keyboard() 
    
    # Show start screen
    call_text(p_win, start_text)
    kb.waitKeys(keyList=[key_dict["pract_key"]])
     
    call_text(p_win, pract_text1)
    kb.waitKeys(keyList=[key_dict["pract_key"]])
    
    call_text(p_win, pract_text2)
    kb.waitKeys(keyList=[key_dict["pract_key"]])
    
    call_text(p_win, pract_text3)
    kb.waitKeys(keyList=[key_dict["pract_key"]])
    
    call_text(p_win, pract_text4)
    kb.waitKeys(keyList=[key_dict["pract_key"]])
    
    call_text(p_win, pract_text5)
    kb.waitKeys(keyList=[key_dict["pract_key"]])
    
        
    if exp_info["enc_type"] == "obj_enc":
        # obj encoding
        enc_pract(p_obj_stim, p_obj_dir, pract_verbs, p_win, 
                  win_time["pract_fix"], win_time["pract_stim"])
        # odd even task
        odd_even(p_win, 8, intro_task, task_text1, task_text2, task_text3, green, red, 
                 key_dict["pract_even"], key_dict["pract_odd"], win_time["task_text"])
        # obj retrieval
        ret_pract(pract_verbs, p_win, win_time["pract_fix"], win_time["pract_stim"],
                   win_time["ret_wait"],)
        
    else:
        # sc encoding
        enc_pract(p_sc_stim, p_sc_dir, pract_adj, p_win, 
                  win_time["pract_fix"], win_time["pract_stim"])
        # odd even task
        odd_even(p_win, 8, intro_task, task_text1, task_text2, task_text3, green, red, 
                 key_dict["pract_even"], key_dict["pract_odd"], win_time["task_text"])
        # sc retrieval
        ret_pract(pract_adj, p_win, win_time["pract_fix"], win_time["pract_stim"],
                   win_time["ret_wait"],)
    
    call_text(p_win, pract_text6)
    kb.waitKeys(keyList=[key_dict["pract_key"]])
    
    call_text(p_win, pract_end)
    kb.waitKeys(keyList=[key_dict["exit_key"]])
    p_win.close()    
    
    sys.exit("Practice Session over. Please re-run script to start with main task.")


#Define block variable from exp_info input
if exp_info["block_no"] == "block_1":
    block=0
elif len(exp_info["block_no"]) == 7 and exp_info["block_no"] != "block_1":
    block= int(exp_info["block_no"][-1:])-1
else:
    block= int(exp_info["block_no"][-2:])-1
    
if exp_info["enc_type"] == "obj_enc":
    matrix_len = obj_matrix.shape[1]
    
    intro_obj = f"""
    Wir beginnen nun mit der richtigen Aufgabe, welche insgesamt aus {obj_matrix.shape[1]//4} aufeinander folgenden Blöcken besteht. 
    \nBitte bleiben Sie während den Messungen möglichst ruhig liegen.
    \n\n>>> Drücken Sie die Taste, wenn Sie bereit sind mit der Aufgabe zu starten."""

else:
    matrix_len = sc_matrix.shape[1]
    
    intro_sc = f"""
    Wir beginnen nun mit der richtigen Aufgabe, welche insgesamt aus {sc_matrix.shape[1]//4} aufeinander folgenden Blöcken besteht. 
    \nBitte bleiben Sie während den Messungen möglichst ruhig liegen.
    \n\n>>> Drücken Sie die Taste, wenn Sie bereit sind mit der Aufgabe zu starten.
    """


#sys.exit("Running of script has been interrupted")   
 
#1920,1080
# Create window
win = visual.Window([1440,900], monitor="testMonitor", fullscr=False, 
                    units='pix', color=light_grey) #lightgrey
win.mouseVisible = False

kb = keyboard.Keyboard() 

if exp_info["enc_type"] == "obj_enc":
    call_text(win, intro_obj)
    kb.waitKeys(keyList=[key_dict["sub_key"]])
else:
    call_text(win, intro_sc)
    kb.waitKeys(keyList=[key_dict["sub_key"]])
fix_cross(win, fix)
print("Subject is ready. Press space and start scanner.")
kb.waitKeys(keyList=[key_dict["operator_key"]])
print("Space was pressed. Waiting for scanner trigger...")
kb.waitKeys(keyList=[key_dict["scanner_key"]])
clock = core.Clock()
    
for i in range(block, matrix_len):
        
    # fixation cross during T1 scan
    if block == matrix_len//2:
        fix_cross(win, fix)
        print("Press 2nd space when T1 completed.")
        kb.waitKeys(keyList=[key_dict["operator_key"]])
        print("Second space received. Continue with task...")
        call_text(win, after_t1)
        kb.waitKeys(keyList=[key_dict["sub_key"]])
        print("Subject is ready. Press space and start scanner.")
        fix_cross(win, fix)
        kb.waitKeys(keyList=[key_dict["operator_key"]])
        print("Space pressed. Waiting for scanner trigger...")
        kb.waitKeys(keyList=[key_dict["scanner_key"]])
        
        
    if exp_info["enc_type"] == "obj_enc":
        block_break = f"Super, Block {(block+1)//4} von insgesamt {obj_matrix.shape[1]//4} Blöcken ist geschafft! \n\nNutzen Sie die Pause, um sich einen kurzen Moment auszuruhen, bevor es mit dem nächsten Block weitergeht. \nDie Pause sollte nicht länger als 5 Minuten dauern. \n\n>>> Drücken Sie die Taste, sobald Sie bereit sind fortzufahren..."
            
        #encoding 
        enc_block(obj_matrix, obj_paths, obj_verbs, t_obj_enc_fix, t_obj_enc, 
                  block, win, ITI_obj_enc, win_time["stim_time"], obj_ext)
     
# =============================================================================
#         # fix break after encoding
#         call_text(win, short_break)
#         core.wait(win_time["enc_break"])
# =============================================================================
        
        # odd even task 
        odd_even(win, 8, intro_task, task_text1, task_text2, task_text3, green, 
                 red, key_dict["even"], key_dict["odd"], win_time["task_text"]) 
        
        #retrieval
        ret_block(obj_matrix, obj_verbs, enc_cat, obj_cat, t_obj_ret_fix, t_obj_ret_cue, 
                   obj_cues, obj_targ, obj_rt, block, win, ITI_obj_ret, 
                   win_time["stim_time"], win_time["ret_wait"], key_dict["sub_key"])
        
    else:
        block_break = f"Super, Block {(block+1)//4} von insgesamt {sc_matrix.shape[1]//4} Blöcken ist geschafft! \n\nNutzen Sie die Pause, um sich einen kurzen Moment auszuruhen, bevor es mit dem nächsten Block weitergeht. \nDie Pause sollte nicht länger als 5 Minuten dauern. \n\n>>> Drücken Sie die Taste, sobald Sie bereit sind fortzufahren..."

        # encoding    
        enc_block(sc_matrix, sc_paths, sc_adj, t_sc_enc_fix, t_sc_enc,
                  block, win, ITI_sc_enc, win_time["stim_time"], sc_ext)  
        
# =============================================================================
#         # fix break after encoding
#         call_text(win, short_break)
#         core.wait(win_time["enc_break"])
# =============================================================================
        
        # odd even task 
        odd_even(win, 8, intro_task, task_text1, task_text2, task_text3, green, 
                 red, key_dict["even"], key_dict["odd"], win_time["task_text"]) 
        
        #retrieval
        ret_block(sc_matrix, sc_adj, sc_paths, sc_paths, t_sc_ret_fix, t_sc_ret_cue, 
                   sc_cues, sc_targ, sc_rt, block, win, ITI_sc_ret, 
                   win_time["stim_time"], win_time["ret_wait"], key_dict["sub_key"])
            
    if (block+1) % 2 == 0:
        fix_cross(win, fix)
        core.wait(win_time["run_fix"])
        
        if block == (matrix_len//2)-1:
            print("Half-time! T1-Scan at beginning of next block. Ask subject to press button if ready for scan and skip break.")
            call_text(win, half_time)
            kb.waitKeys(keyList=[key_dict["sub_key"]])
            print("Subject is ready for scan. Press space and start scanner.")
            kb.waitKeys(keyList=[key_dict["operator_key"]])
            print("First space press received. Start T1 and field maps.")
        else:
            # save main dict to pickle file after each run
            with open(file_path, "wb") as file:
                pickle.dump(main_dict, file, pickle.HIGHEST_PROTOCOL)
                
            # finish experiment if matrix len reached
            if exp_info["enc_type"] == "obj_enc" and block == obj_matrix.shape[1]-1:
                print("Experiment completed.")
            elif exp_info["enc_type"] == "sc_enc" and block == sc_matrix.shape[1]-1:
                print("Experiment completed.")
                
            # individual break with block info after every 4th block 
            elif (block+1)%4 == 0:
                call_text(win, block_break)
                kb.waitKeys(keyList=[key_dict["sub_key"]])
                fix_cross(win, fix)
                print("Subject is ready. Press space and start scanner.")
                kb.waitKeys(keyList=[key_dict["operator_key"]])
                print("Space pressed. Waiting for scanner trigger...")
                kb.waitKeys(keyList=[key_dict["scanner_key"]])
                
            # fixed break after every even block 
            else:
                call_text(win, fix_break2)
                core.wait(win_time["fix_break"]) 
            
    
    elif block == 0:
        call_text(win, fix_break1)
        core.wait(win_time["fix_break"])            
        
    else:
        # fix break after every uneven block number
        call_text(win, fix_break2)
        core.wait(win_time["fix_break"])            
        
    # Count Block numbers for each iteration        
    block = block+1

call_text(win, end_text)
kb.waitKeys(keyList=[key_dict["exit_key"]])
win.close()


# save main dict to pickle file
with open(file_path, "wb") as file:
    pickle.dump(main_dict, file, pickle.HIGHEST_PROTOCOL)
    
# backup
with open(file_path2, "wb") as file:
    pickle.dump(main_dict, file, pickle.HIGHEST_PROTOCOL)

# =============================================================================
# RESTRUCTURE MAIN_DICT
# =============================================================================
# restructure dict based on main_dict mem data from piloting (sub02)
import os
import pickle
import pandas as pd
import numpy as np
import copy

isub = '02'
ises = '1'
itype = 'obj'

# set path to behav_loc file 
main_dict_path = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'MPIB_NeuroCode', 
                            'Sleeplay', 'data_analysis', 'behav', 'behav_loc_data', 'pilots',
                            f"s{isub}", 'behav', f"{itype}", f"ses0{ises}")
path_file = os.path.join(main_dict_path, 'main_dict_s' + isub + '_loc' + ises + '_' + itype + '.pickle')

#load main_dict
data = pd.read_pickle(path_file)

# define encoding categories based on path array
enc_cat_tmp = np.empty(data['obj_paths'].shape, dtype=object)
obj_enc_cat = copy.deepcopy(data['obj_paths'])

# hacky for now, because windows paths can't be split on macOS?
for i,e in np.ndenumerate(obj_enc_cat):
    enc_cat_tmp[i] = e[80:]

runs_per_loc = 12 #10 for scene localiser
trls_per_run = 20 #4x5 trls per block
trls_per_block = 5
trl_count = np.arange(1, (runs_per_loc*trls_per_run)+1)
blocks_per_run = 4
        
label_types = ['enc_fix', 'enc_cue', 'ret_fix', 'ret_cue', 'ret_resp']
#run_nr = 12

# array with all cue_types per run
for i,e in enumerate(label_types):
    labels_tmp = np.repeat(e, trls_per_run)
    if i == 0:
        labels_per_run = labels_tmp
    else:
        labels_per_run = np.hstack((labels_per_run, labels_tmp))

# bring into save_dict shape for all runs
for i in range(runs_per_loc):
    if i == 0:
        cue_type = labels_per_run
    else:
        cue_type = np.hstack((cue_type, labels_per_run))

# create array with run numbers for save_dict
for i in range(1, runs_per_loc+1):
    run_tmp = np.repeat(i, len(labels_per_run))
    if i == 1:
        run_arr = run_tmp
    else:
        run_arr = np.hstack((run_arr, run_tmp))

# create array with block numbers (from 5x48-design)
for i in range(1, (runs_per_loc*blocks_per_run)+1):
    blocks_tmp = np.repeat(i, trls_per_block)
    if i == 1:
        blocks_arr = blocks_tmp
    else:
        blocks_arr = np.hstack((blocks_arr, blocks_tmp))


# define variables for save_dict          
# obj_enc_stim_all = data['enc_obj'].flatten('F')

# ITI_obj = np.hstack((data['ITI_obj_enc'].flatten('F'), data['ITI_obj_ret'].flatten('F')))
# obj_cat = np.hstack((enc_cat_tmp.flatten('F'), data['obj_ret_cat'].flatten('F')))
# obj_ret_targ = data['obj_ret_targ'].flatten('F')
# obj_ret_cue = data['obj_ret_cues'].flatten('F')
# obj_ret_RT = data['obj_ret_rt'].flatten('F')

sub_ID = np.full((cue_type.shape), f"{isub}", dtype=object)
trl_nr = np.full((cue_type.shape), np.nan, dtype=object)
block_nr = np.full((cue_type.shape), np.nan, dtype=object)
run_nr = run_arr
obj_enc_stim = np.full((cue_type.shape), np.nan, dtype=object)
obj_enc_cue = np.full((cue_type.shape), np.nan, dtype=object)
obj_cat = np.full((cue_type.shape), np.nan, dtype=object)
obj_ret_targ = np.full((cue_type.shape), np.nan, dtype=object)
obj_ret_cue = np.full((cue_type.shape), np.nan, dtype=object)
ret_resp = np.full((cue_type.shape), np.nan, dtype=object)
ITIs = np.full((cue_type.shape), np.nan, dtype=object)
time_onset = np.full((cue_type.shape), np.nan, dtype=object)
loc_sess = np.full((cue_type.shape), f"{data['loc_sess'][0]}", dtype=object)


# read in all data for each cue_type
for i,e in enumerate(label_types):
    tmp_idx = (cue_type == e)
    trl_nr[tmp_idx] = trl_count
    block_nr[tmp_idx] = blocks_arr
    if e == 'enc_fix':
        ITIs[tmp_idx] = data['ITI_obj_enc'].flatten('F')
        time_onset[tmp_idx] = data['t_obj_enc_fix'].flatten('F')
    elif e == 'enc_cue':
        obj_enc_stim[tmp_idx] = data['enc_obj'].flatten('F')
        obj_enc_cue[tmp_idx] = data['enc_obj_verbs'].flatten('F')
        obj_cat[tmp_idx] = enc_cat_tmp.flatten('F')
        time_onset[tmp_idx] = data['t_obj_enc'].flatten('F')
    elif e == 'ret_fix':
        ITIs[tmp_idx] = data['ITI_obj_ret'].flatten('F')
        time_onset[tmp_idx] = data['t_obj_ret_fix'].flatten('F')
    elif e == 'ret_cue':
        obj_ret_cue[tmp_idx] = data['obj_ret_cues'].flatten('F')
        obj_ret_targ[tmp_idx] = data['obj_ret_targ'].flatten('F')
        obj_cat[tmp_idx] = data['obj_ret_cat'].flatten('F')
        time_onset[tmp_idx] = data['t_obj_ret_cue'].flatten('F')
    else:
        check_resp = np.asarray(data['obj_ret_rt'].flatten('F'), dtype=float)
        check_resp = ~np.isnan(check_resp)
        ret_resp[tmp_idx] = check_resp
        tmp_RTs = np.array(data['obj_ret_rt'].flatten('F'), dtype=float)
        tmp_RTs = np.nan_to_num(tmp_RTs)
        time_onset[tmp_idx] = data['t_obj_ret_cue'].flatten('F') + tmp_RTs
        
    
save_dict = pd.DataFrame({
    'sub_ID' : sub_ID,
    'trl_nr' : trl_nr,
    'block_nr' : block_nr,
    'run_nr' : run_nr,
    'cue_type': cue_type,
    'obj_enc_stim' : obj_enc_stim,
    'obj_enc_cue' : obj_enc_cue,
    'obj_cat' : obj_cat,
    'obj_ret_targ' : obj_ret_targ,
    'obj_ret_cue' : obj_ret_cue,
    'ret_resp' : ret_resp,
    'ITIs' : ITIs,
    'time_onset' : time_onset,
    'loc_sess' : loc_sess
    })


# # save main dict to pickle file
# with open(file_path, "wb") as file:
#     pickle.dump(main_dict, file, pickle.HIGHEST_PROTOCOL)

# save dict to csv file
# save_name = f"loc_data_sub{isub}.pkl"
save_path = os.path.join(os.sep, 'Users', 'kolbe', 'Documents', 'MPIB_NeuroCode',
                          'Sleeplay', 'data_analysis', 'MRI_loc_data', 'behav')
save_dict.to_csv(os.path.join(save_path, f"loc_{itype}_enc_sub{isub}.csv"), sep='\t', encoding='utf-8', header=True)

# save main dict to pickle file
with open(os.path.join(save_path, f"loc_{itype}_enc_sub{isub}.pkl"), "wb") as file:
    pickle.dump(save_dict, file, pickle.HIGHEST_PROTOCOL)
    

# =============================================================================
# if exp_info["loc_sess"] == "loc1":
#     with open(file_path2, "wb") as file:
#         pickle.dump(main_dict, file, pickle.HIGHEST_PROTOCOL)
# =============================================================================
