#!/bin/bash

lr_assoc_vals=(0.02 0.05)
lr_attn_vals=(0.0033 0.0043)
c_vals=(6.0 6.5)
phi_vals=(2.0 2.5)
datasets=('abstract')
losses=('hinge' 'humble' 'mse' 'll')
epochs=128

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
						python alcove.py -m 'alcove' --lr_assoc $lr_assoc --lr_attn $lr_attn --c $c --phi $phi -d $dataset -l $loss -e $epochs
					done
				done
			done
		done
	done
done