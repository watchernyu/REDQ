#!/bin/sh
#SBATCH --verbose
#SBATCH -p aquila,parallel
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --mem=12GB
#SBATCH --mail-type=ALL # select which email types will be sent
#SBATCH --mail-user=netid@nyu.edu # put your email here if you want emails

#SBATCH --array=0-0 # here the number depends on number of tasks in the array, e.g. 0-59 will create 60 tasks
#SBATCH --output=r_%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID
#SBATCH --error=r_%A_%a.err

# #####################################################
# #SBATCH --gres=gpu:1 # uncomment this line to request a gpu
#SBATCH --constraint=cpu # specify constraint features ('cpu' means only use nodes that have the 'cpu' feature) check features with the showcluster command

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load anaconda3 cuda/9.0 glfw/3.3 gcc/7.3 mesa/19.0.5 llvm/7.0.1
conda init bash
source ~/.bashrc
conda activate /scratch/netid/trial_conda/light_env

echo ${SLURM_ARRAY_TASK_ID}
python train_redq_sac.py --debug --epochs 20