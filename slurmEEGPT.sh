#!/bin/sh

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="EEGPTtest"
#SBATCH --mail-user=chntzi001@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL