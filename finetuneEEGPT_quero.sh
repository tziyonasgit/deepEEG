#!/bin/sh
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="queroEEGPT"
#SBATCH --mail-user=chntzi001@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

module load python/miniconda3-py3.9
source activate /home/chntzi001/.conda/envs/EEGPT_env

# Your science stuff goes below this line...
for fold in {1..5}; do
    python /home/chntzi001/deepEEG/EEGPT/downstream_tueg/run_class_finetuning_EEGPT_change.py \
        --fold $fold \
        --output_dir ./checkpoints/finetune_quero_eegpt/fold_${fold} \
        --log_dir ./log/finetune_quero_eegpt/fold_${fold}  \
        --model EEGPT \
        --finetune /home/chntzi001/deepEEG/EEGPT/downstream_tueg/Checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt \
        --weight_decay 0.05 \
        --batch_size 16 \
        --lr 5e-4 \
        --update_freq 1 \
        --warmup_epochs 5 \
        --epochs 50 \
        --layer_decay 0.65 \
        --dist_eval \
        --save_ckpt_freq 5 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --dataset QUERO \
        --disable_qkv_bias \
        --seed 0            
done