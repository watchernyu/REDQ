#!/bin/sh
#SBATCH --verbose
#SBATCH -p aquila,parallel
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --mem=12GB
#SBATCH --mail-type=ALL # select which email types will be sent
#SBATCH --mail-user=NETID@nyu.edu # NOTE: put your netid here if you want emails

#SBATCH --array=0-0 # here the number depends on number of tasks in the array, e.g. 0-59 will create 60 tasks
#SBATCH --output=r_%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID
#SBATCH --error=r_%A_%a.err

# #####################################################
# #SBATCH --gres=gpu:1 # uncomment this line to request a gpu
#SBATCH --constraint=cpu # specify constraint features ('cpu' means only use nodes that have the 'cpu' feature) check features with the showcluster command

sleep $(( (RANDOM%10) + 1 )) # to avoid issues when submitting large amounts of jobs

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load anaconda3 cuda/9.0 glfw/3.3 gcc/7.3 mesa/19.0.5 llvm/7.0.1 # load modules that we might use
conda init bash # this and the following line are important to avoid hpc issues
source ~/.bashrc
conda activate /scratch/NETID/redq_env # NOTE: remember to change to your actual netid

echo ${SLURM_ARRAY_TASK_ID}
python train_redq_sac.py --debug --epochs 20 --env Hopper-v4 # use v4 version tasks if you followed the newest Gym+MuJoCo installation guide