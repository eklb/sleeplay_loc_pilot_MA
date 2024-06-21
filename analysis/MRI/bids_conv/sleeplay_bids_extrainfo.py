# -*- coding: utf-8 -*-
 #!/usr/bin/env python
# ======================================================================
# SCRIPT INFORMATION:
# ======================================================================
# SCRIPT: UPDATE OF BIDS DIRECTORY
# ======================================================================
# IMPORT RELEVANT PACKAGES
# chmod u+rwx -R .\
# RUN all script on cluster for the right file seperator
# ======================================================================
import glob
import json
import numpy as np
import sys
import os
import stat


sub_list=["01", "02", "03"]
sessions=["01", "02", "03", "04"]

for sub in sub_list:
    for ses in sessions:
        
        path_bids = '/home/mpib/kolbe/sleeplay_tardis/b_loc_v2/data/bids' # tardis data path 
        
        if os.getcwd()[0:12] == '/Users/kolbe':
            path_bids ='/Users/kolbe/Documents/my_GitHub/sleeplay_ma/b_loc_v2/analysis/MRI/bids_conv' # local data path
            

        print(path_bids)

        path_fmap = os.path.join(path_bids,f"sub-{sub}",f"ses-{ses}",'fmap','*.json')
        path_func = os.path.join(path_bids,"sub-{sub}",f"ses-{ses}",'func','*.nii.gz')
        path_desc = os.path.join(path_bids,'dataset_description.json')
        
        # ======================================================================
        # UPDATE DATA-SET DESCRIPTION FILE
        # ======================================================================
        
        # open the dataset_description.json file:
        with open(path_desc) as json_file:
            json_desc = json.load(json_file)
            
        # update fields of the json file:
        json_desc["Name"] = "Sleeplay pilot localiser v2"
        json_desc["Authors"] = ["Marit Petzka", "Elsa Kolbe", "Nicolas W. Schuck"]

        # save updated data-set_description.json file:
        with open(path_desc, 'w') as outfile:
            json.dump(json_desc, outfile, indent=4)
            
        # ======================================================================
        # UPDATE FIELDMAP JSON FILES
        # ======================================================================
        
        # get all fieldmap files in the data-set:
        files_fmap = glob.glob(path_fmap)
        
        # loop over all field-map files:
        for file_path in files_fmap:
            
            with open(file_path,'r') as in_file:
                json_info = json.load(in_file)
                
            in_file.close()
            file_base = os.path.dirname(os.path.dirname(file_path))
            files_func = glob.glob(os.path.join(file_base,'func','*nii.gz'))
            files_func.sort()
            grandparent = os.path.basename(os.path.abspath(os.path.join(file_path,'../..')))
            parent = 'func'
            up_dirs = os.path.join(grandparent, parent)
            
            #intended_for = ['/'.join([parent,'sub-'+str(sub),'ses-'+str(ses), op.basename(file)]) for file in files_func]
            intended_for = [os.path.join(up_dirs, os.path.basename(file)) for file in files_func]
            json_info["IntendedFor"] = intended_for

            # add writing permission to fmap json file
            #os.chmod(file_path,  stat.S_IWGRP | stat.S_IRGRP | stat.S_IRUSR | stat.S_IWUSR | stat.S_IWOTH | stat.S_IROTH)
            os.chmod(file_path, 0o777)
            
            # save updated fieldmap json-file:
            with open(file_path, 'w+') as out_file:
                json.dump(json_info, out_file, indent=2)

