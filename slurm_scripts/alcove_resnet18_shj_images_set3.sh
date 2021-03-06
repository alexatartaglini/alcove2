#!/bin/bash
#
#SBATCH --job-name=shj_alcove_resnet18_shj_images_set3
#SBATCH --output=/home/wv9/code/WaiKeen/alcove2/slurm_logs/shj_alcove_resnet18_shj_images_set3.out
#SBATCH --error=/home/wv9/code/WaiKeen/alcove2/slurm_logs/shj_alcove_resnet18_shj_images_set3.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=12GB
#SBATCH --mail-type=BEGIN,END,FAIL # notifications for job done & fail
#SBATCH --mail-user=waikeenvong@gmail.com

# configuration
model='alcove'
net='resnet18'
dataset='shj_images_set3'
loss='loglik'
epochs=128

# hyperparameters
lr_assoc_vals=(0.01 0.025 0.05)
lr_attn_vals=(0 0.0025 0.005)
c_vals=(2.5 5.0 7.5)
phi_vals=(1.0 2.5 5.0)

source ~/.bashrc
conda activate pytorch
cd /home/wv9/code/WaiKeen/alcove2
    
for lr_assoc in "${lr_assoc_vals[@]}"
do
    for lr_attn in "${lr_attn_vals[@]}"
    do
	for c in "${c_vals[@]}"
	do
	    for phi in "${phi_vals[@]}"	    
            do
                srun python alcove.py --model $model --net $net --dataset $dataset --loss $loss --epochs $epochs --lr_assoc $lr_assoc --lr_attn $lr_attn --c $c --phi $phi --save_results
            done
        done
    done
done
    