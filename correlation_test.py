#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:48:03 2020

@author: Alexa
"""

import scipy
from scipy import stats
from os import mkdir, path
import pandas as pd
import matplotlib.pyplot as plt
import math

from scipy.stats import gaussian_kde

plt.style.use('seaborn-whitegrid')
import numpy as np

class corr_object:
	def __init__(self,s_corr,s_p,k_corr,k_p,correct):
		self.s_corr = s_corr
		self.s_p = s_p
		self.k_corr = k_corr
		self.k_p = k_p
		self.correct = correct
		
def is_correct_ordering(df):
	types = [1,2,3,4,5,6]
	integrals = []
	
	for ty in types:
		integral = df.loc[(df['Type'] == ty)]['Average Integral'].item()
		integrals.append(integral)
	
	order = scipy.stats.rankdata(integrals).tolist()
	correct_orders = [[6,5,4,3,2,1],[6,5,4,2,3,1],[6,5,3,4,2,1],[6,5,3,2,4,1],
				   [6,5,2,3,4,1],[6,5,2,4,3,1]]
	correct = False
	for correct_order in correct_orders:
		if (order == correct_order):
			correct = True
			break
	return correct

def correlation(df):
	# Calculates Spearman's Rho and Kendall's Tau for integrals.
	# Stores correlations in a csv (correlations/correlations.csv).
	#
	# Input
	#	df: a dataframe with integral data (eg. from integrals.csv)
	
	types = [1,2,3,4,5,6]
	ranks = [6, 5, 3, 3, 3, 1]
	integrals = []
	
	for ty in types:
		integral = df.loc[(df['Type'] == ty)]['Average Integral'].item()
		integrals.append(integral)
	
	s_calculation = scipy.stats.spearmanr(ranks,integrals)
	k_calculation = scipy.stats.kendalltau(ranks,integrals)
	
	order = scipy.stats.rankdata(integrals).tolist()
	correct_orders = [[6,5,4,3,2,1],[6,5,4,2,3,1],[6,5,3,4,2,1],[6,5,3,2,4,1],
				   [6,5,2,3,4,1],[6,5,2,4,3,1]]
	correct = False
	for correct_order in correct_orders:
		if (order == correct_order):
			correct = True
			break
	
	correct_csv = pd.DataFrame(index=[0], columns=['Spearman Correlation',
											  'Kendall Correlation','Correct','Order','Integrals'])
	correct_csv.at[0,'Spearman Correlation'] = s_calculation.correlation
	correct_csv.at[0,'Kendall Correlation'] = k_calculation.correlation
	correct_csv.at[0, 'Correct'] = str(correct)
	correct_csv.at[0, 'Order'] = str(order)
	correct_csv.at[0, 'Integrals'] = str(integrals)
	
	correct_dir = 'csv/correlations/correct_correlations.csv'
	try:
		mkdir('csv/correlations')
	except FileExistsError:
		pass
	
	if path.isfile(correct_dir):
		with open(correct_dir, 'a') as csv:
			correct_csv.to_csv(csv, header=False)
	else:
		correct_csv.to_csv(correct_dir)
	
	result = corr_object(s_calculation.correlation, s_calculation.pvalue, 
					  k_calculation.correlation, k_calculation.pvalue,correct)
	
	return result

def df_to_correlation(df):

	for index, row in df.iterrows():
		if not (isinstance(row['Net'], str)) and math.isnan(row['Net']):
			df.at[index,'Net'] = "no"

		if not (isinstance(row['c'], str)) and math.isnan(row['c']):
			df.at[index,'c'] = "no"

	'''
	if(df.at[1,'Net'] == 'nan' or df.at[1,'Net'] is None):
		df['Net'] = 'NaN'
	if(df.at[1,'c'] == 'nan' or df.at[1,'c'] is None):
		df['c'] = 'NaN'
	'''

	args = ['Model', 'Net', 'Loss Type', 'Image Set', 'LR-Attention', 'LR-Association',
							 'c','phi']

	results = df.groupby(args).apply(correlation).reset_index(name='corr')
	print(results)
	cols = args + ['Correct','Spearman Correlation','Spearman p-value','Kendall Correlation','Kendall p-value']
	
	correlations = pd.DataFrame(index=range(0,results.shape[0]), 
								columns=cols)
	
	for index, row in results.iterrows():
		correlations.iloc[index,0:8] = row[0:8]
		corr_object1 = row[8]
		correlations.at[index,'Spearman Correlation'] = corr_object1.s_corr
		correlations.at[index,'Spearman p-value'] = corr_object1.s_p
		correlations.at[index,'Kendall Correlation'] = corr_object1.k_corr
		correlations.at[index,'Kendall p-value'] = corr_object1.k_p
		correlations.at[index,'Correct'] = corr_object1.correct
	
	corr_dir = 'csv/correlations/correlations.csv'
	try:
			mkdir('csv/correlations')
	except FileExistsError:
		pass

	if path.isfile(corr_dir):
		with open(corr_dir, 'a') as csv:
			correlations.to_csv(csv, header=False)
	else:
		correlations.to_csv(corr_dir)

def plot_correlations(df):
	spearman_c = []
	spearman_f = []
	kendall_c = []
	kendall_f = []
	index = 0
	for i, row in df.iterrows():
		correct = bool(row[3])
		s = row[1]
		k = row[2]
		if(correct):
			spearman_c.append(s)
			kendall_c.append(k)
		else:
			spearman_f.append(s)
			kendall_f.append(k)
		
	spearmans = [spearman_c,spearman_f]
	kendalls = [kendall_c,kendall_f]
	colors = ['green','red']
	plt.hlines(min(kendall_c),min(spearman_f),1,linestyles='dashed',label='Min Kendall: ' + str(min(kendall_c)))
	plt.vlines(min(spearman_c),min(kendall_f),1,linestyles='dotted',label='Min Spearman: ' + str(min(spearman_c)))
	plt.legend(loc='lower center')
	plt.xlabel('Spearman Correlation')
	plt.ylabel('Kendall Correlation')
	for spearman, kendall, c in zip(spearmans, kendalls, colors):	
		plt.plot(spearman, kendall, 'o', color=c, alpha=0.02, markersize=5)


def neg_corr(df):
	spearman_vals = df['Spearman Correlation'].tolist()
	configs = []
	configs_full = []
	
	percent_model = [0,0] # alcove, mlp
	percent_net = [0,0,0] # resnet18, resnet152, vgg11
	percent_loss = [0,0,0,0] # hinge, humble, mse, ll
	percent_imageset = [0,0,0,0] # abstract, 1, 2, 3

	for idx in range(0, len(spearman_vals)):
		if spearman_vals[idx] < 0.9 and spearman_vals[idx] > 0.6:
			config = [df.iloc[idx]['Model':'Spearman Correlation']]
			configs_full.append([df.iloc[idx]['Model'], df.iloc[idx]['Net'], df.iloc[idx]['Loss Type'],
								 df.iloc[idx]['Image Set'], df.iloc[idx]['LR-Attention'], df.iloc[idx]['LR-Association'],
								 df.iloc[idx]['c'], df.iloc[idx]['phi']])
			configs.append(config)
			model = df.iloc[idx]['Model']
			if model == 'alcove':
				percent_model[0] = percent_model[0] + 1
			else:
				percent_model[1] = percent_model[1] + 1
			net = df.iloc[idx]['Net']
			if net == 'resnet18':
				percent_net[0] = percent_net[0] + 1
			elif net == 'resnet152':
				percent_net[1] = percent_net[1] + 1
			else:
				percent_net[2] = percent_net[2] + 1
			loss = df.iloc[idx]['Loss Type']
			if loss == 'hinge':
				percent_loss[0] = percent_loss[0] + 1
			elif loss == 'humble':
				percent_loss[1] = percent_loss[1] + 1
			elif loss == 'mse':
				percent_loss[2] = percent_loss[2] + 1
			else:
				percent_loss[3] = percent_loss[3] + 1
			imageset = df.iloc[idx]['Image Set']
			if imageset == 'abstract':
				percent_imageset[0] = percent_imageset[0] + 1
			elif imageset == 'shj_images_set1':
				percent_imageset[1] = percent_imageset[1] + 1
			elif imageset == 'shj_images_set2':
				percent_imageset[2] = percent_imageset[2] + 1
			else:
				percent_imageset[3] = percent_imageset[3] + 1

	'''
	args = ['Model', 'Net', 'Loss Type', 'Image Set', 'LR-Attention', 'LR-Association',
			'c', 'phi']
	corrs = pd.DataFrame(configs_full, columns=args)
	corr_dir = 'csv/correlations/correct_correlations.csv'
	try:
		mkdir('csv/correlations')
	except FileExistsError:
		pass

	if path.isfile(corr_dir):
		with open(corr_dir, 'a') as csv:
			corrs.to_csv(csv, header=False)
	else:
		corrs.to_csv(corr_dir)
	'''

	for i in range(0,len(percent_model)):
		percent_model[i] = str((percent_model[i]*100) / len(configs))
	for i in range(0,len(percent_net)):
		percent_net[i] = str((percent_net[i]*100) / len(configs))
	for i in range(0,len(percent_loss)):
		percent_loss[i] = str((percent_loss[i]*100) / len(configs))
	for i in range(0,len(percent_imageset)):
		percent_imageset[i] = str((percent_imageset[i]*100) / len(configs))


	with open('csv/correlations/almost_correct.txt','w') as f:
		f.write('Total # configs: ' + str(len(configs)) + '\n\n')
		f.write('% ALCOVE: ' + percent_model[0] + '   ' + '% MLP: ' + percent_model[1] + '\n')
		f.write('% resnet18: ' + percent_net[0] + '   ' + '% resnet152: ' + percent_net[1]
			 + '   ' + '% vgg11: ' + percent_net[2] + '\n')
		f.write('% hinge: ' + percent_loss[0] + '   ' + '% humble: ' + percent_loss[1] + '   ' +
			 '% mse: ' + percent_loss[2] + '   ' + '% ll: ' + percent_loss[3] + '\n')
		f.write('% abstract: ' + percent_imageset[0] + '   ' + '% shj_images_set1: ' +
				percent_imageset[1] + '   ' + '% shj_images_set2: ' + percent_imageset[2] +
				'% shj_images_set3: ' + percent_imageset[3] + '\n\n')
		for config in configs:
			for element in config:
				f.write(str(element))
			f.write('\n\n')

	
def histogram(df):
	spearman_range = [] 
	spearman_vals = df['Spearman Correlation'].tolist()
	kendall_range = []
	kendall_vals = df['Kendall Correlation'].tolist()
	
	for s in spearman_vals: 
		if(not (s in spearman_range)):
			spearman_range.append(s)
	spearman_range.sort()
	
	for k in kendall_vals: 
		if(not (k in kendall_range)):
			kendall_range.append(s)
	kendall_range.sort()
	
	spearman_count = [0] * len(spearman_range)
	kendall_count = [0] * len(kendall_range)
	
	for s in spearman_vals:
		idx = spearman_range.index(s)
		spearman_count[idx] = spearman_count[idx] + 1
	
	for k in kendall_vals:
		idx = kendall_range.index(s)
		kendall_count[idx] = kendall_count[idx] + 1
		
	plt.hist(df['Spearman Correlation'], edgecolor='black', bins=20)
	plt.xlabel('Spearman Correlation')
	plt.ylabel('# configurations')
	plt.savefig('figure/hist.png')

def check_contents(df):
	contains_model = [0, 0]
	contains_net = [0, 0, 0]
	contains_imageset = [0, 0, 0, 0, 0]
	contains_loss = [0, 0, 0, 0]

	for index, row in df.iterrows():
		if row['Model'] == 'alcove':
			contains_model[0] += 1
		elif row['Model'] == 'mlp':
			contains_model[1] += 1

		if row ['Net'] == 'none':
			contains_net[0] += 1
		elif row['Net'] == 'resnet18':
			contains_net[1] += 1
		elif row['Net'] == 'resnet152':
			contains_net[2] += 1
		elif row['Net'] == 'vgg11':
			contains_net[3] += 1

		if row['Image Set'] == 'abstract':
			contains_imageset[0] += 1
		elif row['Image Set'] == 'shj_images_set1':
			contains_imageset[1] += 1
		elif row['Image Set'] == 'shj_images_set2':
			contains_imageset[2] += 1
		elif row['Image Set'] == 'shj_images_set3':
			contains_imageset[3] += 1

		if row['Loss Type'] == 'hinge':
			contains_loss[0] += 1
		elif row['Loss Type'] == 'humble':
			contains_loss[1] += 1
		elif row['Loss Type'] == 'mse':
			contains_loss[2] += 1
		elif row['Loss Type'] == 'll':
			contains_loss[3] += 1

def total_vs_num_correct(dfsub):
	total = len(dfsub.index)
	print(total)

	#dfsub = dfsub[dfsub['Correct']]
	dfsub = dfsub[dfsub['Spearman Correlation'] > 0.94] # Threshold Spearman value ~= 0.941
	num_correct = len(dfsub.index)
	if(total != 0):
		percent_correct = (num_correct / total) * 100
	else:
		percent_correct = 0

	return total, num_correct, percent_correct

def percent_correct(df):
	for index, row in df.iterrows():
		if not (isinstance(row['Net'], str)) and math.isnan(row['Net']):
			row['Net'] = "no"
		if not (isinstance(row['c'], str)) and math.isnan(row['c']):
			row['c'] = "no"
			
	models = {'alcove', 'mlp'}
	imagesets = {'abstract', 'shj_images_set1', 'shj_images_set2', 'shj_images_set3'}
	nets = {'resnet18', 'resnet152', 'vgg11_fc'}
	losses = {'hinge', 'humble', 'mse', 'll'}

	strs = []

	for model in models:
		dfsub = df[df['Model'] == model]
		total, num_correct, percent_correct = total_vs_num_correct(dfsub)

		s = "% of " + model + " configurations that achieved correct ordering: " \
			+ str(percent_correct) + "\n" + "(" + str(num_correct) + \
			" nearly correct configs out of " + str(total) + " configs of this kind)\n\n"
		strs.append(s)
		for loss in losses:
			dfsub2 = dfsub[dfsub['Loss Type'] == loss]
			total, num_correct, percent_correct = total_vs_num_correct(dfsub2)

			s = "\t% of " + model + " " + loss + " configurations that achieved correct ordering: " \
				+ str(percent_correct) + "\n\t" + "(" + str(num_correct) + " correct configs out of " \
				+ str(total) + " configs of this kind)\n\n"
			strs.append(s)
			for imageset in imagesets:
				dfsub3 = dfsub2[dfsub2['Image Set'] == imageset]
				total, num_correct, percent_correct = total_vs_num_correct(dfsub3)

				s = "\t\t% of " + model + " " + imageset + " " + loss +  \
					" configurations that achieved correct ordering: " + str(percent_correct) + \
					"\n\t\t" + "(" + str(num_correct) + " correct configs out of " + str(total) \
				    + " configs of this kind)\n\n"
				strs.append(s)
				if(imageset != 'abstract'):
					for net in nets:
						dfsub4 = dfsub3[dfsub3['Net'] == net]

						total, num_correct, percent_correct = total_vs_num_correct(dfsub4)

						s = "\t\t\t% of " + model + " " + imageset + " " + net + " " + loss + \
							" configurations that achieved correct ordering: " + str(percent_correct) \
							+ "\n\t\t\t" + "(" + str(num_correct) + " correct configs out of " + str(total) \
							+ " configs of this kind)\n\n"
						strs.append(s)

	return strs

def count_orderings(df):
	orders = []
	order_spearman = {}

	for index, row in df.iterrows():
		order = row['Order']
		order = order.strip('][').split(', ')
		for i in range(0,6):
			elem = order[i]
			if elem == '6.0':
				order[i] = 1
			elif elem == '5.0':
				order[i] = 2
			elif elem == '4.0':
				order[i] = 3
			elif elem == '3.0':
				order[i] = 4
			elif elem == '2.0':
				order[i] = 5
			else:
				order[i] = 6

		spearman = row['Spearman Correlation']
		if order not in orders:
			orders.append(order)
			order_spearman[str(order)] = [spearman]
		else:
			order_spearman.get(str(order)).append(spearman)

	order_df = pd.DataFrame(index=range(0,len(orders)),columns=['Order', 'Occurrences', 'Average Spearman'])

	index = 0
	for order in orders:
		spearmans = order_spearman.get(str(order))
		occur = len(spearmans)

		average_spearman = sum(spearmans) / occur

		order_df.at[index,'Order'] = order
		order_df.at[index,'Occurrences'] = occur
		order_df.at[index,'Average Spearman'] = average_spearman
		index += 1

	order_dir = 'csv/correlations/orderings.csv'
	try:
		mkdir('csv/correlations')
	except FileExistsError:
		pass

	if path.isfile(order_dir):
		with open(order_dir, 'a') as csv:
			order_df.to_csv(csv, header=False)
	else:
		order_df.to_csv(order_dir)


def plot_density(df):

	x = np.linspace(0, 1, 10)
	spearman_vals = df['Spearman Correlation'].tolist()

	density = gaussian_kde(spearman_vals)
	density.covariance_factor = lambda : .25
	density._compute_covariance()

	plt.plot(x,density(x))
	plt.show()
	'''
	print(df['Spearman Correlation'].tolist())
	#plt.hist(df[:,8], edgecolor='black', bins=30)
	df.hist(column='Spearman Correlation',bins=50)
	plt.xlim(0,1)
	plt.xlabel('Spearman Correlation')
	plt.ylabel('# configurations')
	plt.title(path)
	plt.savefig('figure/density/' + path + '.png')
	'''

def all_densities(df):
	for index, row in df.iterrows():
		if not (isinstance(row['Net'], str)) and math.isnan(row['Net']):
			row['Net'] = "no"
		if not (isinstance(row['c'], str)) and math.isnan(row['c']):
			row['c'] = "no"

	models = {'alcove', 'mlp'}
	imagesets = {'abstract', 'shj_images_set1', 'shj_images_set2', 'shj_images_set3'}
	nets = {'resnet18', 'resnet152', 'vgg11'}
	losses = {'hinge', 'humble', 'mse', 'll'}

	for model in models:
		dfsub = df[df['Model'] == model]
		for loss in losses:
			dfsub2 = dfsub[dfsub['Loss Type'] == loss]
			for imageset in imagesets:
				dfsub3 = dfsub2[dfsub2['Image Set'] == imageset]

				if imageset != 'abstract':
					for net in nets:
						dfsub4 = dfsub3[dfsub3['Net'] == net]
						s = model + "_" + loss + "_" + imageset + "_" + net
						plot_density(dfsub4, s)
				else:
					s = model + "_" + loss + "_" + imageset
					plot_density(dfsub3, s)


def average_corr_by_type(df):
	num_results = 0
	corr_total = 0
	for index, row in df.iterrows():
		num_results += 1
		corr_total += row['Spearman Correlation']
	return corr_total / num_results

#df = pd.read_csv('csv/integrals/integrals.csv')
#df = pd.read_csv('csv/correlations/correct_correlations.csv')
#count_orderings(df)
#df_to_correlation(df)

df = pd.read_csv('csv/correlations/correlations.csv')
for index, row in df.iterrows():
	if not (isinstance(row['Net'], str)) and math.isnan(row['Net']):
		row['Net'] = "no"
	if not (isinstance(row['c'], str)) and math.isnan(row['c']):
		row['c'] = "no"

#result = df.groupby(["Model","Loss Type"]).apply(average_corr_by_type).reset_index(name='Average Correlation')
strs = percent_correct(df)
print(strs)
with open('percent_correct_new.txt', 'w') as f:
    for item in strs:
        f.write("%s" % item)

