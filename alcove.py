import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt    
from itertools import permutations
from data_loader import get_label_coding,load_shj_abstract,load_shj_images
from scipy.stats import sem
from os import mkdir, path
import os
import pandas as pd
import argparse
import scipy.stats


#
# PyTorch implementation of
#  	ALCOVE: An exemplar-based connectionist model of category learning (Kruschke, 1992)

#   Main script runs ALCOVE on stimuli of Shephard, Hovland, and Jenkins (1961)

#  There are a few modifications to Kruschke's original ALCOVE:
#    - with two classes, there is only one binary output
#    - rather the using Kruschke's loss, there are options to maximize
#    the log-likelihood directly (ll loss) or a version of the humble teacher (hinge loss)

class ALCOVE(nn.Module):

	def __init__(self, exemplars, c, phi):
		# Input
		#   exemplars: [ne x dim] rows are exemplars provided to model
		super(ALCOVE, self).__init__()
		self.ne = exemplars.size(0) # number of exemplars
		self.dim = exemplars.size(1) # stimulus dimension
		self.exemplars = exemplars # ne x dim

		# set attention weights to be uniform
		self.attn = torch.nn.Parameter(torch.ones((self.dim,1))/float(self.dim))

		# set association weights to zero
		self.w = torch.nn.Linear(self.ne,1,bias=False)
		self.w.weight = torch.nn.Parameter(torch.zeros((1,self.ne)))

		self.c = c # sharpness parameter (Kruschke uses 6.5 in SHJ simulations)
		self.phi = phi # temperature when making decisions; not included in loss (Kruschke uses 2.0)

	def forward(self,x):
		# Input
		#  x: [dim tensor] a single stimulus
		#
		# Output
		#  output : [tensor scalar] unnormalized log-score (before sigmoid)
		#  prob : [tensor scalar] sigmoid output
		x = x.view(-1,1) # dim x 1
		x_expand = x.expand((-1,self.ne)) # dim x ne
		x_expand = torch.t(x_expand) # ne x dim
		attn_expand = self.attn.expand((-1,self.ne)) # dim x ne
		attn_expand = torch.t(attn_expand) # ne x dim

		# memory/hidden layer is computes the similarity of stimulus x to each exemplar
		hidden = attn_expand * torch.abs(self.exemplars-x_expand) # ne x dim
		hidden = torch.sum(hidden,dim=1) # ne
		hidden = torch.exp(-self.c * hidden) # ne
		hidden = hidden.view((1,-1)) # 1 x ne
		
		# compute the output response
		output = self.w(hidden).view(-1) # tensor scalar
		prob = torch.sigmoid(self.phi*output) # tensor scalar
		return output,prob

class MLP(nn.Module):

	def __init__(self, exemplars, phi, nhid=8):
		# Input
		#   exemplars: [ne x dim] rows are exemplars provided to model
		super(MLP, self).__init__()
		self.ne = exemplars.size(0) # number of exemplars
		self.dim = exemplars.size(1) # stimulus dimension
		self.nhid = nhid
		self.hid = torch.nn.Linear(self.dim,self.nhid)
		self.out = torch.nn.Linear(self.nhid,1)
		self.phi = phi

	def forward(self,x):
		# Input
		#  x: [dim tensor] a single stimulus
		#
		# Output
		#  output : [tensor scalar] unnormalized log-score (before sigmoid)
		#  prob : [tensor scalar] sigmoid output
		x = x.view(1,-1) # dim x 1
		x = self.hid(x)
		x = torch.tanh(x)
		output = self.out(x)
		prob = torch.sigmoid(self.phi*output) # tensor scalar
		return output,prob

class corr_object:
	
	def __init__(self,s_corr,s_p,k_corr,k_p):
		self.s_corr = s_corr
		self.s_p = s_p
		self.k_corr = k_corr
		self.k_p = k_p

def update_batch(net,exemplars,targets,loss,optimizer):
	# Update the weights using batch SGD for the entire set of exemplars
	#
	# Input
	#   exemplars: [ne x dim tensor] all stimuli/exempalrs in experiment 
	#   targets:   [ne tensor] classification targets (1/0 or 1/-1, depending on loss)
	#   loss: function handle
	#   optimizer : SGD optimizer
	net.zero_grad()
	net.train()
	n_exemplars = exemplars.size(0)
	out = torch.zeros(n_exemplars)
	for j in range(n_exemplars):
		out[j],_ = net.forward(exemplars[j])
	myloss = loss(out, targets)
	myloss.backward()
	optimizer.step()
	if model_type == 'alcove':
		net.attn.data = torch.clamp(net.attn.data, min=0.) # ensure attention is non-negative
	return myloss.cpu().item()

def update_single(net, exemplars, targets, loss, optimizer):
	#net.zero_grad()
	#net.train()
	torch.autograd.set_detect_anomaly(True)
	
	n_exemplars = exemplars.size(0)
	out = torch.zeros(n_exemplars)
	current_loss = torch.zeros(n_exemplars)
	for j in range(n_exemplars):
		net.zero_grad()
		net.train()
	
		out[j],_ = net.forward(exemplars[j])
		myloss = loss(out[j], targets[j])
		myloss.backward(retain_graph=True)
		optimizer.step()
		current_loss[j] = myloss.cpu().item()
		if model_type == 'alcove':
			net.attn.data = torch.clamp(net.attn.data, min=0.) # ensure attention is non-negative
	return torch.sum(current_loss).item()
	

def evaluate(net,exemplars,targets):
	# Compute probability of getting each answer/exemplar right using sigmoid
	# 
	# Input
	#   exemplars: [ne x dim tensor] all stimuli/exempalrs in experiment 
	#   targets:   [ne tensor] classification targets (1/0 or 1/-1, depending on loss)
	#
	# Output
	#   mean probability of correct response
	#   mean accuracy when picking most likely response
	net.eval()
	n_exemplars = exemplars.size(0)
	v_acc = np.zeros(n_exemplars)
	v_prob = np.zeros(n_exemplars)
	for j in range(n_exemplars):
		out,prob = net.forward(exemplars[j])
		out = out.item() # logit
		prob = prob.item() # prob of decision 
		if targets[j].item()==POSITIVE:		
			v_prob[j] = prob
			v_acc[j] = out >= 0	
		elif targets[j].item()==NEGATIVE:
			v_prob[j] = 1-prob
			v_acc[j] = out < 0
	return np.mean(v_prob), 100.*np.mean(v_acc)

def HingeLoss(output, target):
	# Reinterpretation of Kruschke's humble teacher
	#  loss = max(0,1-output * target)
	#
	# Input
	#  output : 1D tensor (raw prediction signal)
	#  target : 1D tensor (must be -1. and 1. labels)
    hinge_loss = 1.-torch.mul(output, target)
    hinge_loss[hinge_loss < 0] = 0.
    return torch.sum(hinge_loss)

def HumbleTeacherLoss(output, target):
	humble_loss = torch.mul(output, target)
	humble_loss[humble_loss > 1] = 1
	humble_loss = (1.-humble_loss)**2
	return .5 * torch.sum(humble_loss)


def train(exemplars,labels,num_epochs,loss_type,typenum,c,phi,df,track_inc=5,verbose_params=False):
	# Train model on a SHJ problem
	# 
	# Input
	#   exemplars : [n_exemplars x dim tensor] rows are exemplars
	#   labels : [n_exemplars tensor] category labels	
	#   num_epochs : number of passes through exemplar set
	#   loss_type : either 'll' or 'hinge'
	#	track_inc : track loss/output at these intervals
	#   verbose_params : print parameters when you are done
	#
	# Output
	#    trackers for epoch index, probability of right response, accuracy, and loss
	#    each is a list with the same length
	n_exemplars = exemplars.size(0)
	
	if model_type == 'mlp':
		net = MLP(exemplars,phi)
	elif model_type == 'alcove':
		net = ALCOVE(exemplars,c,phi)
	else:
		assert False

	if loss_type == 'll':
		loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
	elif loss_type == 'hinge':
		loss = HingeLoss
	elif loss_type == 'mse':
		loss = torch.nn.MSELoss(reduction='sum')
	elif loss_type == 'humble':
		loss = HumbleTeacherLoss
	else:
		assert False # undefined loss
	
	optimizer = optim.SGD(net.parameters(), lr=lr_association)
	if model_type == 'alcove':
		optimizer = optim.SGD([ {'params': net.w.parameters()}, {'params' : [net.attn], 'lr':lr_attn}], lr=lr_association)

	v_epoch = []
	v_loss = []
	v_acc = []
	v_prob = []
	for epoch in range(1,num_epochs+1):
		#loss_epoch = update_batch(net,exemplars,labels,loss,optimizer)
		loss_epoch = update_single(net,exemplars,labels,loss,optimizer)
		print(loss_epoch)
		if epoch == 1 or epoch % track_inc == 0:
			test_prob,test_acc = evaluate(net,exemplars,labels)
			v_epoch.append(epoch)
			v_loss.append(loss_epoch / float(n_exemplars))
			v_acc.append(test_acc)
			v_prob.append(test_prob)
			print('  epoch ' + str(epoch) + "; train loss " + str(round(v_loss[-1],4)))
			if(track_inc == 1):
				df.at[(epoch//track_inc+type_tracker)-1, 'Train Loss'] = round(v_loss[-1],4)
				df.at[(epoch//track_inc+type_tracker)-1, 'Train Accuracy'] = round(v_acc[-1],4)
				df.at[(epoch//track_inc+type_tracker)-1, 'Probability Correct'] = round(v_prob[-1],4)
				df.at[(epoch//track_inc+type_tracker)-1, 'Epoch'] = epoch
			else:
				df.at[epoch//track_inc+type_tracker, 'Train Loss'] = round(v_loss[-1],4)
				df.at[epoch//track_inc+type_tracker, 'Train Accuracy'] = round(v_acc[-1],4)
				df.at[epoch//track_inc+type_tracker, 'Probability Correct'] = round(v_prob[-1],4)
				df.at[epoch//track_inc+type_tracker, 'Epoch'] = epoch

	if model_type == 'alcove' and verbose_params:
		print("Attention weights:")
		print(np.transpose(np.round(net.attn.data.numpy(),3)))
		print("Category associations:")
		print(np.round(net.w.weight.data.numpy(),3))

	return v_epoch,v_prob,v_acc,v_loss

def initialize_df(track_inc,num_rows,list_exemplars,args):
	# Initializes a DataFrame for translation to .csv
	
	num_epochs = args[4]
	epoch_range = list(range(track_inc,num_epochs+1,track_inc))
	epoch_range.insert(0,1)
	df = pd.DataFrame(index=range(1,(num_rows)*ntype*len(list_exemplars)+1), 
				      columns=['Model','Net','Loss Type','Image Set','LR-Attention',
				   'LR-Association','c','phi','Permutation','Type','Epoch','Max Epochs','Train Loss',
				   'Train Accuracy','Probability Correct'])
	df.at[:,'Model'] = args[0]
	df.at[:,'Image Set'] = args[1]
	df.at[:,'Net'] = args[2]
	df.at[:,'Loss Type'] = args[3]
	df.at[:,'LR-Association'] = args[5]
	df.at[:,'LR-Attention'] = args[6]
	df.at[:,'c'] = args[7]
	df.at[:,'phi'] = args[8]
	
	return df

def create_dir(model_type,image_set,net_type,loss_type,_num_epochs,plot):
	# Creates directory and file names for plot/csv storage
	if(image_set == 'abstract'):
		d = 'ab'
		sub_subdir_name = ''
	else:
		d = 'im'
		sub_subdir_name = f'/{image_set}'
		
	if(plot):
		dir_name = 'plots'
		subdir_name = f'{dir_name}/{model_type}_{d}'
		sub_subdir_name = subdir_name + sub_subdir_name
		file_dir = f'{sub_subdir_name}/{net_type}{loss_type}_{num_epochs}' 
		title = f'{model_type} model: {d} stimulus_{net_type}{loss_type}'
	else:
		dir_name = 'csv'
		subdir_name = f'{dir_name}/{model_type}_{d}'
		file_dir = f'{subdir_name}/{net_type}{loss_type}_{num_epochs}' 
		title = None
		
	try:
		mkdir(dir_name)
	except FileExistsError:
		pass
	try:
		mkdir(subdir_name)
	except FileExistsError:
		pass
	if(plot):
		try:
			mkdir(sub_subdir_name)
		except FileExistsError:
			pass
		
	return file_dir,title

def create_plot(list_trackers,ntype,title,file_dir):
	A = np.array(list_trackers) # nperms x ntype x 4 tracker types x n_iters
	M = np.mean(A,axis=0) # ntype x 4 tracker types x n_iters
	SE = sem(A,axis=0) # ntype x 4 tracker types x n_iters		
				
	plt.figure(1)
	for i in range(ntype): 
		if viz_se:
			plt.errorbar(M[i,0,:],M[i,1,:],yerr=SE[i,1,:],linewidth=4./(i+1))
		else:
			plt.plot(M[i,0,:],M[i,1,:],linewidth=4./(i+1))
								
	plt.suptitle(title)
	plt.xlabel('Block')
	plt.ylabel('Probability correct')
	plt.legend(["Type " + str(s) for s in range(1,7)])
	plt.savefig(file_dir + '1.png')
				
	plt.figure(2)
	for i in range(ntype):
		if viz_se:
			plt.errorbar(M[i,0,:],M[i,3,:],yerr=SE[i,3,:],linewidth=4./(i+1))  # v is [tracker type x n_iters]
		else:
			plt.plot(M[i,0,:],M[i,3,:],linewidth=4./(i+1))  # v is [tracker type x n_iters]
				
	plt.suptitle(title)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend(["Type " + str(s) for s in range(1,7)])
	plt.savefig(file_dir + '2.png')
	plt.show()
	
def trapezoidal_integral(f, a = 0, b = 128, step = 1):
	# Calculates an approximate integral from x=a to x=b 
	# using the trapezoidal rule. 
	#
	# Input
	#	f: list of y values/datapoints corresponding to range x=(a,b)
	#	a: starting point/epoch (usually 0) 
	#	b: ending point/epoch (usually 128)
	#	step: step size between points/epochs (usually 1)
	#
	# Output
	#	trapezoidal approximation of integral	
	
	n = (b - a) / step
	delta_x = (b - a) / n
	mul = delta_x / 2
	
	b -= 1 # for index purposes
	t = f[a] + f[b]
	
	for i in range(1,b):
		t += (f[i] * 2)
		
	return t * mul

def average_integral(df, track):
	# Calculates the average approximate integral over permutations.
	
	perms = [0,1,2,3,4,5]
	average = 0

	for perm in perms:
		this_df = df.loc[df['Permutation'] == perm]
		
		f = this_df['Probability Correct'].tolist()
		integral = trapezoidal_integral(f, step=track)
		
		average += integral
		
	return average / len(perms)

def df_to_integral(df, track_inc):
	# Calculates average integrals from dataframe. Saves
	# in csv file (titled integrals.csv)
	
	if(df.at[1,'Net'] == 'nan' or df.at[1,'Net'] is None):
		df['Net'] = 'NaN'
	if(df.at[1,'c'] == 'nan' or df.at[1,'c'] is None):
		df['c'] = 'NaN'

	args = ['Model', 'Net', 'Loss Type', 'Image Set', 'LR-Attention', 'LR-Association',
							 'c','phi','Type']
	
	integrals = df.groupby(args).apply(average_integral,track=track_inc).reset_index(name='Average Integral')
	integral_dir = 'csv/integrals/integrals.csv'
	
	try:
		mkdir('csv/integrals')
	except FileExistsError:
		pass
	
	if path.isfile(integral_dir):
		with open(integral_dir, 'a') as csv:
			csv.write('\n')
			integrals.to_csv(csv, header=False)
	else:
		integrals.to_csv(integral_dir)
		
	df_to_correlation(integrals)

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
	
	return corr_object(s_calculation.correlation, s_calculation.pvalue, 
					  k_calculation.correlation, k_calculation.pvalue)
		
def df_to_correlation(df):
	if(df.at[1,'Net'] == 'nan' or df.at[1,'Net'] is None):
		df['Net'] = 'NaN'
	if(df.at[1,'c'] == 'nan' or df.at[1,'c'] is None):
		df['c'] = 'NaN'
		
	args = ['Model', 'Net', 'Loss Type', 'Image Set', 'LR-Attention', 'LR-Association',
							 'c','phi']
	
	results = df.groupby(args).apply(correlation).reset_index(name='corr')
	cols = args + ['Correct','Spearman Correlation','Spearman p-value','Kendall Correlation','Kendall p-value']
	
	correlations = pd.DataFrame(index=range(0,results.shape[0]), 
								columns=cols)
	
	correlations.iloc[0,0:8] = results.iloc[0,0:8]
	corr_object1 = results.iloc[0,8]
	
	correlations.at[0,'Spearman Correlation'] = corr_object1.s_corr
	correlations.at[0,'Spearman p-value'] = corr_object1.s_p
	correlations.at[0,'Kendall Correlation'] = corr_object1.k_corr
	correlations.at[0,'Kendall p-value'] = corr_object1.k_p
	correlations.at[0,'Correct'] = is_correct_ordering(df)
	
	corr_dir = 'csv/correlations/correlations.csv'
	try:
			mkdir('csv/correlations')
	except FileExistsError:
		pass
		
	if path.isfile(corr_dir):
		with open(corr_dir, 'a') as csv:
			csv.write('\n')
			correlations.to_csv(csv, header=False)
	else:
		correlations.to_csv(corr_dir)		
	
def run_simulation(model_type,image_set,net_type,loss_type,num_epochs,lr_association,lr_attn,c,phi,
				   track_inc,plot):
	im_dir = 'data/' + image_set # assuming imagesets in a folder titled data
	
	# counters for proper translation to .csv file
	global type_tracker
	type_tracker = 1
	global num_rows 
	if(track_inc == 1):
		num_rows = (num_epochs // track_inc)
	else:
		num_rows = (num_epochs // track_inc) + 1
	perm_tracker = 1
	num_rows_p = num_rows*ntype
	
	list_perms = list(permutations([0,1,2])) # ways of assigning abstract dimensions to visual ones
	list_exemplars = []
	for p in list_perms:
		if(image_set == 'abstract'):
			exemplars,labels_by_type = load_shj_abstract(loss_type,p)
		else:
			exemplars,labels_by_type = load_shj_images(loss_type,net_type,im_dir,p)
		# [n_exemplars x dim tensor],list of [n_exemplars tensor]
		list_exemplars.append(exemplars)
	
	# initialize DataFrame for translation to .csv
	df = initialize_df(track_inc,num_rows,list_exemplars,[model_type, image_set, net_type, 
		  loss_type, num_epochs,lr_association,lr_attn,c,phi])
	
	dim = list_exemplars[0].size(1)
	print("Data loaded with " + str(dim) + " dimensions.")
	
	# Run ALCOVE on each SHJ problem
	list_trackers = []
	for pidx,exemplars in enumerate(list_exemplars): # all permutations of stimulus dimensions
		tracker = []
		print('Permutation ' + str(pidx))
		df.at[perm_tracker:perm_tracker+num_rows_p,'Permutation'] = pidx
		for mytype in range(1,ntype+1): # from type I to type VI
			print('  Training on type ' + str(mytype))
			df.at[type_tracker:type_tracker+num_rows,'Type'] = mytype
			df.at[type_tracker:type_tracker+num_rows,'Max Epochs'] = num_epochs
			labels = labels_by_type[mytype-1]
			v_epoch,v_prob,v_acc,v_loss = train(exemplars,labels,num_epochs,loss_type,mytype,c,phi,df,track_inc)
			tracker.append((v_epoch,v_prob,v_acc,v_loss))
			print("")
			type_tracker += num_rows
		list_trackers.append(tracker)
		perm_tracker += num_rows_p
		
	# create directories/filenames for plots/.csv files and title for plots	
	if(image_set == 'abstract'):		
		file_dir,title = create_dir(model_type,image_set,'',loss_type,num_epochs,plot)
	else:
		file_dir,title = create_dir(model_type,image_set,net_type+'_',loss_type,num_epochs,plot)

	# plot or save to .csv
	if(plot):
		create_plot(list_trackers,ntype,title,file_dir)
	else:      
		if(path.isfile(file_dir + '.csv')):
			with open(file_dir + '.csv', 'a', newline='\n') as csv:
				csv.write('\n')
				df.to_csv(csv, header=False)
		else:
			df.to_csv(file_dir + '.csv')
	
	# Calculate and store average integrals for this setting
	df_to_integral(df, track_inc)
	
if __name__ == "__main__":
	
	os.environ['KMP_DUPLICATE_LIB_OK']='True'
	
	# create argparse arguments for simulation parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("-m","--model" ,help="Model (alcove or mlp)",type=str,default='alcove')
	parser.add_argument("-d","--dataset",help="Dataset (shj_images_set1, abstract, etc)",type=str,default='shj_images_set1')
	parser.add_argument("-n","--net",help="Net type (resnet18, resnet152, vgg11",type=str,default='resnet18')
	parser.add_argument("-l","--loss",help="Loss (hinge, ll, mse, humble)",type=str,default='humble')
	parser.add_argument("-e","--epochs",help="# Epochs (default 128)",type=int,default=128)
	parser.add_argument("--lr_assoc",help="Learning rate - association (default 0.03)",type=float,default=0.03)
	parser.add_argument("--lr_attn",help="Learning rate - attention (default 0.0033)",type=float,default=0.0033)
	parser.add_argument("--c",help="c hyperparameter(default 6.5)",type=float,default=6.5)
	parser.add_argument("--phi",help="phi hyperparameter (default 2.0)",type=float,default=2.0)
	
	args = parser.parse_args()
	
	model_type = args.model
	image_set = args.dataset
	net_type = args.net
	loss_type = args.loss
	num_epochs = args.epochs
	lr_association = args.lr_assoc
	lr_attn = args.lr_attn
	c = args.c
	phi = args.phi
	
	ntype = 6 # number of types in SHJ
	viz_se = False # visualize standard error in plot	

	# options for output of results
	plot = False # saves plots when true
	track_inc = 1 # step size for recording epochs
	POSITIVE,NEGATIVE = get_label_coding(loss_type)
	
	# create directory for extracted features 
	try:
		mkdir('pickle')
	except FileExistsError:
		pass

	print(f'config: {model_type}, {image_set}, {net_type}, {loss_type} loss, {num_epochs} epochs')

	# run simulation
	if(model_type == 'alcove' and image_set != 'abstract'): # data type = images, model = alcove
		run_simulation(model_type,image_set,net_type,loss_type,num_epochs,lr_association,lr_attn,c,phi,
				 track_inc,plot)
					
	elif(model_type == 'alcove'): # data type = abstract, model = alcove	
		run_simulation(model_type,image_set,None,loss_type,num_epochs,lr_association,lr_attn,c,phi,
				 track_inc,plot)
				
	elif(model_type == 'mlp' and image_set != 'abstract'): # data type = images, model = mlp 
		run_simulation(model_type,image_set,net_type,loss_type,num_epochs,lr_association,lr_attn,None,phi,
				 track_inc,plot)
				
	else: 	# data type = abstract, model = mlp	
		run_simulation(model_type,image_set,None,loss_type,num_epochs,lr_association,lr_attn,None,phi,
				 track_inc,plot)

