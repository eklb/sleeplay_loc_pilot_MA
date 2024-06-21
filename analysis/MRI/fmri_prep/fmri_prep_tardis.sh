#!/bin/bash
# ==============================================================================
# SCRIPT INFORMATION:
# ==============================================================================
# SCRIPT: RUN FMRIPREP ON THE MPIB CLUSTER (TARDIS)
# PROJECT: STATE FORMATION

# ==============================================================================
# DEFINE ALL PATHS:
# ==============================================================================
# path to repo folder:
PATH_REPO="$HOME/sleeplay_ma"
# path to project data folder on Tardis:
PATH_DATA="$HOME/sleeplay_tardis"
# define name of current task:
TASK_NAME="fmriprep"
# path to the fmriprep singularity image (predefined on Tardis):
PATH_CONTAINER="/mnt/beegfs/container/fmriprep/fmriprep-22.0.0.sif"
# path to freesurfer license file on tardis:
#PATH_FS_LICENSE="$HOME/.license"
PATH_FS_LICENSE="/opt/freesurfer/7.0.0/.license" # has to be specified accordingly 
# path to data directory (bids format):
PATH_BIDS="${PATH_DATA}/b_loc_v2/data/bids"
# path to output directory to save preprocessed data in:
PATH_OUT="${PATH_DATA}/b_loc_v2/data/derivatives" # output of fmri prep
# path to freesurfer output directory:
PATH_FREESURFER="${PATH_OUT}/freesurfer"
# path to working directory:
PATH_WORK="${PATH_OUT}/work"
# path to log directory:
PATH_LOG="${PATH_DATA}/b_loc_v2/logs/fmriprep"
# define path for the templateflow cache
PATH_TEMPLATEFLOW="$HOME/.templateflow"
# define the path for the fmriprep cache:
PATH_CACHE_FMRIPREP="$HOME/.cache"

# ==============================================================================
# CREATE RELEVANT DIRECTORIES:
# ==============================================================================
mkdir -p ${PATH_OUT}
mkdir -p ${PATH_FREESURFER}
mkdir -p ${PATH_WORK}
mkdir -p ${PATH_LOG}
mkdir -p ${PATH_TEMPLATEFLOW}
mkdir -p ${PATH_CACHE_FMRIPREP}

# ==============================================================================
# DEFINE PARAMETERS:
# ==============================================================================
# maximum number of cpus per process:
N_CPUS=8
# maximum number of threads per process:
N_THREADS=8
# memory demand in *GB*
MEM_GB=30
# memory demand in *MB*
MEM_MB="$((${MEM_GB} * 1000))"

#participants=(01 02 03 04 05 06 07 08 10 11 12 13) # 204)
sub_list=(1 2 3)

# ==============================================================================
# RUN FMRIPREP:
# ==============================================================================
# loop over all subjects:
for sub in ${sub_list[*]}; do
	# get the subject number with zero padding:
	SUB_PAD=$(printf "%02d\n" ${sub})
	# skip fmriprep for input data that does not exist (yet):
	if [ ! -d "${PATH_BIDS}/sub-${SUB_PAD}" ]; then
		echo "BIDS input data for sub-${SUB_PAD} does not exist! Skipping ..."
		continue
	fi
	# create participant-specific working directory:
	PATH_WORK_SUB="${PATH_WORK}/sub-${SUB_PAD}"
	mkdir -p ${PATH_WORK_SUB}
	
	# create a new job file:
	echo "#!/bin/bash" > job
	# name of the job
	echo "#SBATCH --job-name fmriprep_fs_sub-${SUB_PAD}" >> job
	# add partition to job
	echo "#SBATCH --partition long" >> job
	# set the expected maximum running time for the job:
	echo "#SBATCH --time 80:00:00" >> job
	# determine how much RAM your operation needs:
	echo "#SBATCH --mem ${MEM_GB}GB" >> job
	# write log to log folder
	echo "#SBATCH --output ${PATH_LOG}/slurm_fmriprep_sub-${SUB_PAD}_%j.out" >> job
	# request multiple cpus
	echo "#SBATCH --cpus-per-task ${N_CPUS}" >> job
	# export template flow environment variable:
	echo "export SINGULARITYENV_TEMPLATEFLOW_HOME=/templateflow" >> job 

	echo "singularity run --cleanenv --contain \
	-B ${PATH_FS_LICENSE}:/.license:ro \
	-B ${PATH_BIDS}:/input:ro \
	-B ${PATH_OUT}:/output:rw \
	-B ${PATH_FREESURFER}:/output/freesurfer:rw \
	-B ${PATH_WORK_SUB}:/work:rw \
	-B ${PATH_TEMPLATEFLOW}:/templateflow:rw \
	-B ${PATH_CACHE_FMRIPREP}:/cache \
	${PATH_CONTAINER} \
	--fs-license-file /.license --notrack \
	/input/ /output/ participant --participant_label ${SUB_PAD} -w /work/ \
	--mem_mb ${MEM_MB} --nthreads ${N_CPUS} --omp-nthreads ${N_THREADS} \
	--write-graph --stop-on-first-crash \
	--output-spaces T1w MNI152NLin6Asym fsnative fsaverage\
	--notrack --verbose " >> job
	# submit job to cluster queue and remove it to avoid confusion:
	sbatch job
	rm -f job
done
