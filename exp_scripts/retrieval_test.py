# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:01:02 2023

@author: elsak
"""

from psychopy import visual, core, gui
from psychopy.hardware import keyboard
import numpy as np
import random
import sys
import os
import itertools
import datetime
import pickle
from instructions_retrieval_test import *


# =============================================================================
# Define all directories
# =============================================================================
#C:\Users\elsak\Documents\GitHub\sleeplay_fMRI_b\tempus\stimuli
#C:\Users\neuroadmin\Desktop\Marit\sleeplay_fMRI_b\tempus\stimuli
base_dir = os.path.join('C:\\', 'Users', 'elsak', 'OneDrive', 'Dokumente', 'GitHub', 
   'sleeplay_fMRI_b', 'tempus', 'stimuli')

obj_dir = os.path.join(base_dir, 'images', 'objects')
obj_cat = os.listdir(obj_dir) # list of all object categories from folder
sc_dir = os.path.join(base_dir, 'images', 'scenes')


# =============================================================================
# Define dicts for screen times and keys
# =============================================================================
win_time = dict()
win_time["fix_cross"] = 4
win_time["max_wait"] = 5
win_time["exit_time"] = 2

key_dict = dict()
key_dict["sub_key"] = "space"
key_dict["exit_key"] = "escape"

# =============================================================================
# Define all functions for experiment
# =============================================================================

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
        save_file(main_dict, exp_info, all_cues, all_targets, react_times, date)
        # save retrieval dict to pickle file 
        with open(file_path, "wb") as file:
            pickle.dump(retrieval_dict, file, pickle.HIGHEST_PROTOCOL)
            
        call_text(win, exit_text)
        kb.waitKeys(keyList=["escape"])
        win.close()
        core.quit()

def extract_values(main_dict, seed, r_start, r_end):
    """
    Function to extract cues and targets from main_dict of given sub_ID for retrieval test.
    
    Passed arguments:
        main_dict = dict of sub_ID from localiser
        seed = random shuffle_idx 
        r_start = r_start
        r_end = r_end
        
    """
    global cues, targets
    cues, targets = list(), list()
    
    for i in range(r_start, r_end):
        if main_dict["loc_sess"][0][:1] == "o":
            cues.append(list(main_dict["enc_obj_verbs"][:,i]))
            targets.append(list(main_dict["enc_obj"][:,i]))
        else:
            cues.append(list(main_dict["enc_sc_adj"][:,i]))
            targets.append(list(main_dict["enc_sc"][:,i]))
            
    # extract all values into one list
    cues = list(itertools.chain(*cues))
    targets = list(itertools.chain(*targets))
    
    # use same shuffle index for cues and targets to keep pairings
    rdm_state = np.random.RandomState(seed)
    rdm_state.shuffle(cues)
    rdm_state.seed(seed)
    rdm_state.shuffle(targets)

def save_file(main_dict, exp_info, all_cues, all_targets, react_times, date):
#save file to new dict
    global retrieval_dict 
    retrieval_dict = dict()
    retrieval_dict["sub_ID"] = exp_info["sub_ID"]
    if main_dict["loc_sess"][0][:1] == "o":
        retrieval_dict["enc_type"] = "obj_enc"
        retrieval_dict["obj_ret_cues"] = all_cues
        retrieval_dict["obj_ret_targ"] = all_targets
    else:
        retrieval_dict["enc_type"] = "sc_enc"
        retrieval_dict["sc_ret_cues"] = all_cues
        retrieval_dict["sc_ret_targ"] = all_targets
    retrieval_dict["targ_inp"] = target_input
    retrieval_dict["rt"] = react_times
    retrieval_dict["date"] = date.strftime("%d/%m/%Y/%H:%M")

        
# =============================================================================
# Create dialogue box to load main_dict of given Sub_ID
# =============================================================================

exp_info = {"sub_ID" : "sXX", # sXX preferred
            "loc_sess" : ["loc1", "loc2"],
            "enc_type" : ["obj_enc", "sc_enc"]} 
exp_gui = gui.DlgFromDict(exp_info, 
                         title="Please fill in the following fields:",
                         labels={"sub_ID" : "Enter Subject_ID \n(e.g. s01):",
                                 "loc_sess" : "1st or 2nd Localiser Session?",
                                 "enc_type" : "Object or Scene encoding?"},
                         order=["sub_ID", "loc_sess", "enc_type"])
                         
#Check whether main dict file exists for sub ID input
if exp_info["enc_type"] == "obj_enc":
    # main_dict path from scanner session 
    main_dict_path = os.path.join(base_dir, "data", f"{exp_info['sub_ID']}", f"main_dict_{exp_info['sub_ID']}_{exp_info['loc_sess']}_obj.pickle")
    # path to save retrieval dict 
    file_path = os.path.join(base_dir, "data", f"{exp_info['sub_ID']}", f"retrieval_dict_{exp_info['sub_ID']}_{exp_info['loc_sess']}_obj.pickle")
else:
    # main_dict path from scanner session 
    main_dict_path = os.path.join(base_dir, "data", f"{exp_info['sub_ID']}", f"main_dict_{exp_info['sub_ID']}_{exp_info['loc_sess']}_sc.pickle")
    # path to save retrieval dict 
    file_path = os.path.join(base_dir, "data", f"{exp_info['sub_ID']}", f"retrieval_dict_{exp_info['sub_ID']}_{exp_info['loc_sess']}_sc.pickle")

file_exist = os.path.exists(main_dict_path)

#Save actual date and time 
date = datetime.datetime.today()


if file_exist:
    with open(main_dict_path, "rb") as file:
        main_dict = pickle.load(file)
        
# Create empty np arrays 
if main_dict["loc_sess"][0][:1] == "o":
    all_cues = np.empty(main_dict["enc_obj_verbs"].shape[0]*main_dict["enc_obj_verbs"].shape[1]//2,
                        dtype=object)
    all_targets = np.empty(main_dict["enc_obj"].shape[0]*main_dict["enc_obj"].shape[1]//2,
                        dtype=object)
    enc_matrix = main_dict["enc_obj"]
else:
    all_cues = np.empty(main_dict["enc_sc_adj"].shape[0]*main_dict["enc_sc_adj"].shape[1]//2,
                        dtype=object)
    all_targets = np.empty(main_dict["enc_sc"].shape[0]*main_dict["enc_sc"].shape[1]//2,
                        dtype=object)
    enc_matrix = main_dict["enc_sc"]


react_times = np.empty(len(all_cues), dtype=object)
target_input = np.empty(len(all_cues), dtype=object)

# Define text stimulus based on length of all_cues matrix
text1 = f"""Die Abfrage wird insgesamt {len(all_cues)} Wörter umfassen und ist in 4 Blöcke gegliedert. Zwischen jedem Block wird es die Möglichkeit zu einer Pause geben, in der Sie sich kurz ausruhen können.
\nDrücken Sie dann, wie gehabt, die Leertaste, sobald Sie bereit sind mit dem nächsten Block fortzufahren.
\n\n>>> Weiter >>>
"""
# Extract and shuffle retrieval cues from main_dict for both halfs separately
# Create random shuffle index, same for cues and targets
seed = np.random.randint(0,100)

# extract first half of cues and targets
extract_values(main_dict, seed, 0, enc_matrix.shape[1]//4)

# fill arrays 
all_cues[:len(all_cues)//2] = cues
all_targets[:len(all_targets)//2] = targets

# extract second half of cues and targets
extract_values(main_dict, seed, enc_matrix.shape[1]//4, enc_matrix.shape[1]//2)

all_cues[len(all_cues)//2:len(all_cues)] = cues
all_targets[len(all_targets)//2:len(all_targets)] = targets

#sys.exit("Script was interrupted.")

# =============================================================================
# Start experiment
# =============================================================================      

win = visual.Window([1920,1080], monitor="testMonitor", fullscr=True, 
                    units='pix', color=(0.654901960784314, 
                                        0.654901960784314, 
                                        0.654901960784314)) #lightgrey
win.mouseVisible = False

kb = keyboard.Keyboard()

call_text(win, title_text)
kb.waitKeys(keyList=[key_dict["sub_key"]])

call_text(win, intro_text)
kb.waitKeys(keyList=[key_dict["sub_key"]])

call_text(win, intro_text2)
kb.waitKeys(keyList=[key_dict["sub_key"]])

call_text(win, text1)
kb.waitKeys(keyList=[key_dict["sub_key"]])

call_text(win, text2)
kb.waitKeys(keyList=[key_dict["sub_key"]])

for i in range(len(all_cues)): 
    exit_exp(win)
    print(f"Retrieval Test Trial {i+1} of {len(all_cues)}")
    
    if i == len(all_cues)*1//4:
        call_text(win, break_text1)
        kb.waitKeys(keyList=[key_dict["sub_key"]])
        
    if i == len(all_cues)//2:
        call_text(win, break_text2)
        kb.waitKeys(keyList=[key_dict["sub_key"]])
        
    if i == len(all_cues)*3//4:
        call_text(win, break_text3)
        kb.waitKeys(keyList=[key_dict["sub_key"]])
        
    fix_cross(win, fix)
    core.wait(win_time["fix_cross"])
    
    word_cue = visual.TextStim(win, text=all_cues[i], color=(-1,-1,-1), 
                               pos=(0,win.size[1]*0.125), height=52) 
    q_mark = visual.TextStim(win, text="?", color=(-1,-1,-1), height=52)
    word_cue.draw()
    q_mark.draw()
    kb.clock.reset()
    win.flip()
    
    key_press = kb.waitKeys(maxWait=win_time["max_wait"], keyList=[key_dict["sub_key"]], waitRelease=True)

    if key_press is None:
        react_times[i] = np.nan
        target_input[i] = ""
    else:
        react_times[i] = key_press[0].rt
        word_cue = visual.TextStim(win, text=all_cues[i], color=(-1,-1,-1), 
                                   pos=(0,win.size[1]*0.125), height=52) 
        q_mark = visual.TextStim(win, text="?", color=(-0.529411764705882, 
                                                       0.403921568627451, 
                                                       -0.113725490196078), #mediumseagreen
                                 height=52)
        word_cue.draw()
        q_mark.draw()
        win.flip()
        core.wait(win_time["max_wait"]-key_press[0].rt)
        
        word_title = visual.TextStim(win, text=input_text, color=(-1,-1,-1), 
                                   pos=(0,win.size[1]*0.125), height=52, wrapWidth=1400) 
        word_input = visual.TextStim(win, height=52, color=(-1,-1,-1))
        keyboardKeys = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z', 'space']
        answer = ' '
        
        word_input.setText(f"{answer}".format(answer))
        word_title.draw()
        word_input.draw()
    
        win.flip()
    
        # get some keys
        kb = keyboard.Keyboard()
        kb.clearEvents(eventType=('keyboard'))
        x = True
        while x == True:
            keys = kb.getKeys()
            for key in keyboardKeys:
                if key in keys:
                    if 'space' in keys:
                        key = ' '
                    answer += key
                    word_input.setText(f"{answer}")
                    word_title.draw()
                    word_input.draw()
                    win.flip()
                    
                if 'backspace' in keys:
                    answer = ''
                    word_input.setText(f"{answer}")
                    word_title.draw()
                    word_input.draw()
                    win.flip()
                    
                if 'escape' in keys:
                    win.close()
                    # save retrieval dict to pickle file 
                    save_file(main_dict, exp_info, all_cues, all_targets, react_times, date)
                    with open(file_path, "wb") as file:
                        pickle.dump(retrieval_dict, file, pickle.HIGHEST_PROTOCOL)
                    core.quit()
                            
                if 'return' in keys:
                    x = False
        target_input[i]= f"{answer}"
        
    if i == len(all_cues):
        call_text(win, end_text)
        kb.waitKeys(keyList=[key_dict["exit_key"]])
        win.close()

# =============================================================================
# Create new dict to save all data 
# =============================================================================

# save retrieval dict to pickle file 
save_file(main_dict, exp_info, all_cues, all_targets, react_times, date)

with open(file_path, "wb") as file:
    pickle.dump(retrieval_dict, file, pickle.HIGHEST_PROTOCOL)
    

   