#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=90
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --account=training2321
#SBATCH --output=%j.out
#SBATCH --error=%j.err

##SBATCH --reservation=ai_sc_day2

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

source $HOME/course/$USER/sc_venv_template/activate.sh

srun python to_h5.py