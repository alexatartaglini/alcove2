# script to create separate slurm scripts to run various simulations

import itertools

# alcove simulations
model = 'alcove'
nets = ['vgg16', 'resnet18', 'resnet50']
datasets = ['shj_images_set1', 'shj_images_set2', 'shj_images_set3']
configurations = list(itertools.product(nets, datasets))

for net, dataset in configurations:
    slurm_str = f"""#!/bin/bash
#
#SBATCH --job-name=shj_alcove_{net}_{dataset}
#SBATCH --output=/home/wv9/code/WaiKeen/alcove2/slurm_logs/shj_alcove_{net}_{dataset}.out
#SBATCH --error=/home/wv9/code/WaiKeen/alcove2/slurm_logs/shj_alcove_{net}_{dataset}.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=12GB
#SBATCH --mail-type=BEGIN,END,FAIL # notifications for job done & fail
#SBATCH --mail-user=waikeenvong@gmail.com

# configuration
model='alcove'
net='{net}'
dataset='{dataset}'
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
    """

    slurm_str += """
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
    """
        
    with open(f'slurm_scripts/alcove_{net}_{dataset}.sh', 'w') as f:
        f.write(slurm_str)
