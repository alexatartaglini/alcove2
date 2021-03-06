import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from convnet_feat import get_features, imshow
import matplotlib.pyplot as plt
import pickle
from os import path

# Functions for loading SHJ abstract data and images

# Filenames for SGJ stimuli and labels in abstract form
fn_shj_abstract = 'data/shj_stimuli.txt'
fn_shj_labels = 'data/shj_labels.txt'


def get_label_coding(loss_type):
    # Set coding for class A and class B
    POSITIVE = 1.
    if loss_type == 'hinge':
        NEGATIVE = -1.
    elif loss_type == 'loglik':
        NEGATIVE = 0.
    elif loss_type == 'mse':
        NEGATIVE = 0.
    elif loss_type == 'humble':
        NEGATIVE = -1.
    else:
        assert False
    return POSITIVE, NEGATIVE


def load_shj(loss_type):
    # Loads SHJ data from text files
    #
    # Input
    #   loss_type : either ll or hinge loss
    #
    # Output
    #   X : [ne x dim tensor] stimuli as rows
    #   y_list : list of [ne tensor] labels, with a list element for each shj type
    stimuli = pd.read_csv(fn_shj_abstract, header=None).to_numpy()
    labels = pd.read_csv(fn_shj_labels, header=None).to_numpy()
    stimuli = stimuli.astype(float)
    ntype = labels.shape[0]
    POSITIVE, NEGATIVE = get_label_coding(loss_type)
    labels_float = np.zeros(labels.shape, dtype=float)
    labels_float[labels == 'A'] = POSITIVE
    labels_float[labels == 'B'] = NEGATIVE
    X = torch.tensor(stimuli).float()
    y_list = []
    for mytype in range(ntype):
        y = labels_float[mytype, :].flatten()
        y = torch.tensor(y).float()
        y_list.append(y)
    return X, y_list


def process_shj_images(net_type, im_dir):
    # Return
    #  X : [ne x dim tensor] stimuli as rows

    pickle_dir_stim = 'pickle/' + net_type + '_' + im_dir[5:] + 'stim.pickle'
    pickle_dir_im = 'pickle/' + net_type + '_' + im_dir[5:] + 'im.pickle'

    if(path.exists(pickle_dir_stim) and path.exists(pickle_dir_im)):
        infile = open(pickle_dir_stim, 'rb')
        stimuli = pickle.load(infile)
        infile.close()

        infile = open(pickle_dir_im, 'rb')
        images = pickle.load(infile)
        infile.close()
    else:
        print("Passing SHJ images through ConvNet...")
        # stimuli,images = get_features(mydir,'vgg11')
        stimuli, images = get_features(im_dir, net_type)

        print("Done.")

        stimuli = stimuli.cpu().data.numpy().astype(float)
        images = images.cpu().data.numpy()
        outfile = open(pickle_dir_stim, 'wb')
        pickle.dump(stimuli, outfile)
        outfile.close()

        outfile = open(pickle_dir_im, 'wb')
        pickle.dump(images, outfile)
        outfile.close()

    X = torch.tensor(stimuli).float()
    return X, images


def load_shj_abstract(loss_type, perm=[0, 1, 2]):
    # Loads SHJ data in abstract form
    #
    # Input
    #   loss_type : either ll or hinge loss
    #   perm : permutation of abstract feature indices
    #
    # Output
    #   X : [ne x dim tensor] stimuli as rows
    #   y_list : list of [ne tensor] labels, with a list element for each shj type

    # load image and abstract data
    X, y_list = load_shj(loss_type)
    X_abstract = X.data.numpy().astype(int)

    # Apply permutation
    X_perm = X_abstract.copy()
    X_perm = X_perm[:, perm]  # permuted features
    perm_idx = []
    for x in X_perm:
        # get item mapping from original order to perm order
        idx = np.where((X_abstract == x).all(axis=1))[0]
        perm_idx.append(idx[0])
    perm_idx = np.array(perm_idx)
    X = X[perm_idx, :]  # permute items from original order to permuted order
    return X, y_list


def load_shj_images(loss_type, net_type, im_dir, perm=[0, 1, 2], viz_cats=False):
    # Loads SHJ data from images
    #
    # Input
    #   loss_type : either ll or hinge loss
    #   perm : permutation of abstract feature indices
    #
    # Output
    #   X : [ne x dim tensor] stimuli as rows
    #   y_list : list of [ne tensor] labels, with a list element for each shj type

    # load image and abstract data
    X, images = process_shj_images(net_type, im_dir)
    X_abstract, y_list = load_shj(loss_type)
    X_abstract = X_abstract.data.numpy().astype(int)

    # Apply permutation
    X_perm = X_abstract.copy()
    X_perm = X_perm[:, perm]  # permuted features
    perm_idx = []
    for x in X_perm:
        # get item mapping from original order to perm order
        idx = np.where((X_abstract == x).all(axis=1))[0]
        perm_idx.append(idx[0])
    perm_idx = np.array(perm_idx)
    X = X[perm_idx, :]  # permute items from original order to permuted order
    # permute items from original order to permuted order
    images = images[perm_idx]

    if viz_cats:
        for mytype in range(6):  # for each type
            y = y_list[mytype].data.numpy()
            images_A = images[y == 1.]
            images_B = images[y != 1.]
            plt.figure(mytype+1)
            for j in range(len(images_A)):
                ax = plt.subplot(2, 4, j+1)
                ax.axis('off')
                imshow(images_A[j])
            for j in range(len(images_B)):
                ax = plt.subplot(2, 4, 5+j)
                ax.axis('off')
                imshow(images_B[j])
            # plt.savefig(str(mytype) + '.pdf')
        plt.show()

    return X, y_list


def load_shj_PCA(loss_type, net_type, im_dir):
    fn_PCA = 'data/PCA/PCA_' + net_type + '_' + im_dir[11:] + '.txt'
    stimuli = pd.read_csv(fn_PCA, header=None).to_numpy()
    labels = pd.read_csv(fn_shj_labels, header=None).to_numpy()
    stimuli = stimuli.astype(float)
    ntype = labels.shape[0]
    POSITIVE, NEGATIVE = get_label_coding(loss_type)
    labels_float = np.zeros(labels.shape, dtype=float)
    labels_float[labels == 'A'] = POSITIVE
    labels_float[labels == 'B'] = NEGATIVE
    X = torch.tensor(stimuli).float()
    y_list = []
    for mytype in range(ntype):
        y = labels_float[mytype, :].flatten()
        y = torch.tensor(y).float()
        y_list.append(y)
    return X, y_list


def load_shj_abstract_PCA(loss_type, net_type, im_dir, perm=[0, 1, 2]):
    # Loads SHJ data in abstract form
    #
    # Input
    #   loss_type : either ll or hinge loss
    #   perm : permutation of abstract feature indices
    #
    # Output
    #   X : [ne x dim tensor] stimuli as rows
    #   y_list : list of [ne tensor] labels, with a list element for each shj type

    # load image and abstract data
    X, y_list = load_shj_PCA(loss_type, net_type, im_dir)
    X_abstract, y_list = load_shj(loss_type)
    X_abstract = X_abstract.data.numpy().astype(int)

    # Apply permutation
    X_perm = X_abstract.copy()
    X_perm = X_perm[:, perm]  # permuted features
    perm_idx = []
    for x in X_perm:
        # get item mapping from original order to perm order
        idx = np.where((X_abstract == x).all(axis=1))[0]
        perm_idx.append(idx[0])
    perm_idx = np.array(perm_idx)
    X = X[perm_idx, :]  # permute items from original order to permuted order
    return X, y_list
