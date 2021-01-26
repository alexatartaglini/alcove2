#!/bin/bash

models=('resnet152')
lr_assoc_vals=(0.03)
lr_attn_vals=(0.0033)
c_vals=(6.5)
phi_vals=(2.0)
datasets=('PCA_abstract')
losses=('ll')

epochs=128

# hinge, c=7.0, lr_assoc=0.03, lr_attn=0.0023, phi=2.5, all datasets
# then all settings except lr_assoc=0.04

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
