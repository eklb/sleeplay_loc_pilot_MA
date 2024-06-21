#!/bin/bash
#HEURISTIC_PATH="$HOME/sleeplay_ma/b_loc_v2/analysis/MRI/bids_conv/sleeplay_heudiconv_heuristic.py" #folder containing scripts
LOGS_PATH="$HOME/sleeplay_tardis/b_loc_v2/logs/bids/"	# Path where error and out logs of cluster jobs are saved
#CODE_PATH="$HOME/SeqTest/heudiconv" # path to all code

sub_list=('02' '03')
sessions=('01' '02')
enc_type=('OBJ' 'SC') 

#SUB_COUNT=0 # initalize a SUBject counter

for sub in ${sub_list[@]}; do # loop over all subjects
	for enc in ${enc_type[@]}; do # both encoding types
    	for ses in ${sessions[@]}; do # 2 sessions per enc type 
    	
    		INPUT_PATH="$HOME/sleeplay_tardis/b_loc_v2/data/raw/SLEEPLAY_V2_PILOT${sub}_${enc}_PILOT${sub}/*/*SES-${ses}*/*IMA" #for subjects 201-203 (STIMPILOT1)
    		OUTPUT_PATH="$HOME/sleeplay_tardis/b_loc_v2/data/bids"
    		echo "sub${sub}"
    		echo $enc
    		echo "ses${ses}"
    		if [ $enc == "OBJ" ]; then 
            	ses1=$ses
            	HEURISTIC_PATH="$HOME/sleeplay_ma/b_loc_v2/analysis/MRI/bids_conv/sleeplay_heudiconv_heuristic_obj.py" #script for obj naming
            else
                HEURISTIC_PATH="$HOME/sleeplay_ma/b_loc_v2/analysis/MRI/bids_conv/sleeplay_heudiconv_heuristic_sc.py" #script for sc naming
                if [ $ses == "01" ]; then
                    ses1="03"
                else
                    ses1="04"
                fi
            fi
    		echo '#!/bin/bash'                             > job.slurm
    		echo "#SBATCH --job-name conv2bids"  		>> job.slurm
    		echo "#SBATCH --partition quick"               >>   job.slurm #only form smaller/quicker jobs
    		echo "#SBATCH --time 1:0:0"                   >> job.slurm #2:0:0
    		echo "#SBATCH --mem 16GB"                      >> job.slurm #16GB
    		echo "#SBATCH --cpus-per-task 1 "              >> job.slurm
    		echo "#SBATCH --output ${LOGS_PATH}\logbids_sub${sub}_ses${ses}_${enc}.out"   >> job.slurm
    		echo "#SBATCH --mail-type NONE"                >> job.slurm
    		#echo "source /etc/bash_completion.d/virtualenvwrapper" >> job.slurm
    		#echo "workon neurogrid" >> job.slurm
    		echo "heudiconv --files ${INPUT_PATH} -s ${sub} -ss ${ses1} -f ${HEURISTIC_PATH} -c dcm2niix -o ${OUTPUT_PATH} -b --overwrite"  >> job.slurm # run the job
    		sbatch job.slurm
    		rm -f job.slurm
    		
    	done
    done
done