#!/bin/bash
#
#SBATCH --job-name=shj_mlp
#SBATCH --output=/home/wv9/code/WaiKeen/alcove2/slurm_logs/shj_mlp.out
#SBATCH --error=/home/wv9/code/WaiKeen/alcove2/slurm_logs/shj_mlp.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=12GB
#SBATCH --mail-type=BEGIN,END,FAIL # notifications for job done & fail
#SBATCH --mail-user=waikeenvong@gmail.com

# configuration
model='mlp'
nets=('vgg16' 'resnet18' 'resnet50')
datasets=('shj_images_set1' 'shj_images_set2' 'shj_images_set3')
loss='loglik'
epochs=128

# hyperparameters
lr_assoc_vals=(0.01 0.025 0.05)
phi_vals=(1.0 2.5 5.0)

source ~/.bashrc
conda activate pytorch
cd /home/wv9/code/WaiKeen/alcove2

for net in "${nets[@]}"
    do
	for dataset in "${datasets[@]}"
	do
            for lr_assoc in "${lr_assoc_vals[@]}"
            do
	        for phi in "${phi_vals[@]}"
                do
                srun python alcove.py --model $model --net $net --dataset $dataset --loss $loss --epochs $epochs --lr_assoc $lr_assoc --phi $phi --save_results
            done
        done
    done
done
    
