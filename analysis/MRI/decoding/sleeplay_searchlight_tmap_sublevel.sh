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
PATH_SCRIPT="${PATH_REPO}/b_loc_v2/analysis/MRI/decoding/sleeplay_searchlight_tmap_sublevel.py"
# path to the log directory:
PATH_LOG="${PATH_BASE}/${PROJECT_NAME}/b_loc_v2/logs/searchlight" 


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
N_CPUS=2
# maximum number of threads per process:
N_THREADS=2
# memory demand in *GB*
MEM_GB=25
# memory demand in *MB*
MEM_MB="$((${MEM_GB} * 1000))"
# create subject list
# SUB_LIST=('01' '02' '03')
# # user-defined subject list
# PARTICIPANTS=$1
# # Get participants to work on
# cd ${PATH_BIDS}
# SUB_LIST=sub-*
# # Only overwrite sub_list with provided input if not empty
# if [ ! -z "${PARTICIPANTS}" ]; then
#   echo "Specific participant ID supplied"
#   # Overwrite sub_list with supplied participant
#   SUB_LIST=${PARTICIPANTS}
# fi

# =============================================================================
# RUN DECODING
# =============================================================================
# loop over all subjects:

#echo "sub-${SUB}"
# get the subject number with zero padding:
#SUB_PAD=$(printf "%02d\n" ${SUB})
# Get job name
JOB_NAME="sleeplay_searchlight_tmaps" #SUB_PAD
# Create job file
echo "#!/bin/bash" > job.slurm
# name of the job
echo "#SBATCH --job-name ${JOB_NAME}" >> job.slurm
# set the expected maximum running time for the job:
echo "#SBATCH --time 12:00:00" >> job.slurm
# determine how much RAM your operation needs:
#echo "#SBATCH --mem ${MEM_GB}GB" >> job.slurm
# memory per CPUs
echo "#SBATCH --mem-per-cpu ${MEM_GB}GB" >> job.slurm 
# determine number of CPUs
echo "#SBATCH --cpus-per-task ${N_CPUS}" >> job.slurm
# write to log folder
echo "#SBATCH --output ${PATH_LOG}/log_searchlight_tmaps.%j.out" >> job.slurm

# Load virtual env
echo "source ${PATH_BASE}/sleeplay_tardis/sleeplay_venv/bin/activate" >> job.slurm 
# echo "source /etc/bash_completion.d/virtualenvwrapper" >> job.slurm
# echo "workon sleeplay_env" >> job.slurm
echo "python3 ${PATH_SCRIPT}" >> job.slurm 
# submit job to cluster queue and remove it to avoid confusion:
sbatch job.slurm
rm -f job.slurm



