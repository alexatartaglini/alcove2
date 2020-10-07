#!/bin/bash

models=('resnet18' 'resnet152' 'vgg11')
lr_assoc_vals=(0.02 0.03 0.04)
lr_attn_vals=(0.0023 0.0033 0.0043)
c_vals=(6.0 6.5 7.0)
phi_vals=(1.5 2.5)
datasets=('shj_images_set1' 'shj_images_set2' 'shj_images_set3')
losses=('hinge' 'humble' 'mse' 'll')
epochs=128

for model in "${models[@]}"
do
	for lr_assoc in "${lr_assoc_vals[@]}"
	do
		for lr_attn in "${lr_attn_vals[@]}"
		do
			for c in "${c_vals[@]}"
			do
				for phi in "${phi_vals[@]}"
				do
					for dataset in "${datasets[@]}"
					do
						for loss in "${losses[@]}"
						do
							python alcove.py -m 'alcove' -n $model --lr_assoc $lr_assoc --lr_attn $lr_attn --c $c --phi $phi -d $dataset -l $loss -e $epochs
						done
					done
				done
			done
		done
	done
done
