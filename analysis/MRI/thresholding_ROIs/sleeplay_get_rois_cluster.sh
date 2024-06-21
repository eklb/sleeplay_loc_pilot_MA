#!/usr/bin/bash
# ==============================================================================
# ==============================================================================
# DEFINE ALL PATHS:
# ==============================================================================
#path to home directory:
PATH_HOME="${HOME}"
# path to the project directories:
PATH_PROJECT="${PATH_HOME}/sleeplay_ma"
PATH_DATA="${PATH_HOME}/sleeplay_tardis" 
PATH_VENV="${PATH_DATA}/sleeplay_venv"

# cd into the directory of the current script:
#cd ${PATH_CODE}
# path to the current script:
PATH_SCRIPT="${PATH_PROJECT}/b_loc_v2/analysis/MRI/thresholding_ROIs/sleeplay_get_rois.py"
#${PATH_CODE}/getThresholdedROI.py
# path to the output directory:
PATH_OUT="${PATH_DATA}/b_loc_v2/data/masks"
# path to the working directory:
#PATH_WORK=${PATH_HOME}/${PROJECT_NAME}/${TASK_NAME}/work
# path to the log directory:
PATH_LOG="${PATH_DATA}/b_loc_v2/logs/masks" # /logs/$(date '+%Y%m%d_%H%M%S')

# ==============================================================================
# CREATE RELEVANT DIRECTORIES:
# ==============================================================================
# create directory for log files:
if [ ! -d ${PATH_LOG} ]; then
    mkdir -p ${PATH_LOG}
fi
# mkdir -p ${PATH_LOG}

# define variables
SUB_LIST=('01' '02' '03') #('01' '02' '03')
#sessions=('03') #('01' '02' '03' '04')
MEM_GB=8

# ==============================================================================
# RUN THE SCRIPT:
# ==============================================================================

for SUB in ${SUB_LIST[@]}; do

	# create additional subject folder for files
	PATH_SUB_FILES="${PATH_OUT}/sub-${SUB}" 
	if [ ! -d ${PATH_SUB_FILES} ]; then
        mkdir -p ${PATH_SUB_FILES}
    fi
	#mkdir -p ${PATH_SUB_FILES}

		echo '#!/bin/bash'                             > job.slurm
		echo "#SBATCH --job-name sleeplay-ROIdropout-sub-${SUB}"  >> job.slurm
		#echo "#SBATCH --partition quick"               >>   job.slurm
		echo "#SBATCH --time 1:0:0"                   >> job.slurm
		echo "#SBATCH --mem ${MEM_GB}GB"                      >> job.slurm
		echo "#SBATCH --cpus-per-task 1 "              >> job.slurm
		echo "#SBATCH --output ${PATH_LOG}_ROI_${SUB}_OutPut.out"   >> job.slurm
		echo "#SBATCH --mail-type NONE"                >> job.slurm
		# Load virtual env
    	echo "source ${PATH_VENV}/bin/activate" >> job.slurm 
		# echo "source /etc/bash_completion.d/virtualenvwrapper" >> job.slurm
		# echo "workon ${PATH_VENV}" >> job.slurm
		echo "python3 ${PATH_SCRIPT} $SUB" >> job.slurm
		sbatch job.slurm
		rm -f job.slurm
	#done
done
