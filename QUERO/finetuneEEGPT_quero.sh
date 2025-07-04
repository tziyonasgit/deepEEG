#!/bin/sh
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="queroEEGPT"
#SBATCH --mail-user=chntzi001@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

module load python/miniconda3-py3.9
source /opt/exp_soft/miniconda3-py3.9/etc/profile.d/conda.sh
conda activate EEGPT_env

# Your science stuff goes below this line...
    python /home/chntzi001/deepEEG/EEGPT/downstream_tueg/run_class_finetuning_EEGPT_change.py \
        --eval  \
        --output_dir ./checkpoints/finetune_quero_eegpt/test \
        --log_dir ./log/finetune_quero_eegpt/test  \
        --model EEGPT \
        --finetune /home/chntzi001/deepEEG/checkpoints/finetune_quero_eegpt/fold_4/checkpoint-best.pth \
        --weight_decay 0.05 \
        --batch_size 16 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --dataset QUERO \
        --disable_qkv_bias \
        --seed 0            
