#!/bin/sh

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="EEGPTtest"
#SBATCH --mail-user=chntzi001@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

module load python/miniconda3-py3.9

source activate /home/chntzi001/.conda/envs/EEGPT_env

python /home/chntzi001/deepEEG/EEGPT/downstream_tueg/dataset_maker/make_TUAB.py