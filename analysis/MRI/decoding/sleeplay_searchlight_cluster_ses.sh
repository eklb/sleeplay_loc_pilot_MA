#!/usr/bin/bash
# -*- coding: utf-8 -*-

# =============================================================================
# DEFINE IMPORTANT PATHS
# =============================================================================

# define home directory
PATH_BASE="/home/mpib/${USER}"
# define the name of the project:
PROJECT_NAME="sleeplay_tardis"
# Define path to repository
PATH_REPO="${PATH_BASE}/sleeplay_ma"
#PATH_BIDS="${PATH_REP}/bids"
# Define path to script to run
PATH_SCRIPT="${PATH_REPO}/b_loc_v2/analysis/MRI/decoding/sleeplay_searchlight_ses.py"
# path to the log directory:
PATH_LOG="${PATH_BASE}/${PROJECT_NAME}/b_loc_v2/logs/searchlight"
# define path for output files
# PATH_OUT="${}"


# =============================================================================
# CREATE DIRECTORIES
# =============================================================================
# # create output directory:
# if [ ! -d ${PATH_OUT} ]; then
#     mkdir -p ${PATH_OUT}
# fi

# create directory for log files:
if [ ! -d ${PATH_LOG} ]; then
	mkdir -p ${PATH_LOG}
fi

# =============================================================================
# DEFINE JOB PARAMETERS FOR CLUSTER
# =============================================================================

# maximum number of cpus per process:
N_CPUS=10
# maximum number of threads per process:
N_THREADS=2
# memory demand in *GB*
MEM_GB=10
# memory demand in *MB*
MEM_MB="$((${MEM_GB} * 1000))"
# create subject list
SUB_LIST=('01' '02' '03')
SES_LIST=('01' '02' '03' '04')


# =============================================================================
# RUN DECODING
# =============================================================================
# loop over all subjects:
for SUB in ${SUB_LIST[@]}; do

    if [ "$SUB" = "03" ]; then
    # Define a new variable
        SES_LIST=('02' '03' '04')
    fi
    
    for SES in ${SES_LIST[@]}; do
    
        #echo "sub-${SUB}"
        # get the subject number with zero padding:
    	#SUB_PAD=$(printf "%02d\n" ${SUB})
    	# Get job name
    	JOB_NAME="sleeplay_searchlight_sub-${SUB}_ses-${SES}" #SUB_PAD
    	# Create job file
    	echo "#!/bin/bash" > job.slurm
    	# name of the job
    	echo "#SBATCH --job-name ${JOB_NAME}" >> job.slurm
    	# set the expected maximum running time for the job:
    	echo "#SBATCH --partition long" >> job.slurm 
    	echo "#SBATCH --time 120:00:00" >> job.slurm 
    
    	# echo "#SBATCH --time 12:00:00" >> job.slurm
    
    	# determine how much RAM your operation needs:
    	#echo "#SBATCH --mem ${MEM_GB}GB" >> job.slurm
    	# memory per CPUs
    	echo "#SBATCH --mem-per-cpu ${MEM_GB}GB" >> job.slurm 
    	# determine number of CPUs
    	echo "#SBATCH --cpus-per-task ${N_CPUS}" >> job.slurm
    	# write to log folder
    	echo "#SBATCH --output ${PATH_LOG}/log_searchlight_sub-${SUB}_ses-${SES}.%j.out" >> job.slurm
    
    	# Load virtual env
    	echo "source ${PATH_BASE}/sleeplay_tardis/sleeplay_venv/bin/activate" >> job.slurm 
        # echo "source /etc/bash_completion.d/virtualenvwrapper" >> job.slurm
        # echo "workon sleeplay_env" >> job.slurm
    	echo "python3 ${PATH_SCRIPT} '${SUB}' '${SES}'" >> job.slurm 
    	# submit job to cluster queue and remove it to avoid confusion:
        sbatch job.slurm
        rm -f job.slurm
        
    done

done

