import numpy as np
from numpy import random
import torch
from matplotlib import pyplot as plt
from cmath import inf
from scipy import stats
from matplotlib import cm
import pandas as pd
from utils_basic import *
import warnings
""" Author: Kaining Zhang, 05/04/2025 """

def similarity_measurement(pattern1, pattern2):
    assert pattern1.shape[-1] == pattern2.shape[-1], "Two pattern's dimensions do not equal"
    assert pattern1.ndim<3 and pattern2.ndim<3, "Can't calculate patterns with dimension >=3"
    n_neuron = pattern1.shape[-1]
    if pattern2.ndim==2:
        pattern2 = pattern2.transpose()

    similarity = np.dot(pattern1*2-1, pattern2*2-1)/n_neuron
    return similarity

def make_scatter_plot(ax, xdata:dict, ydata:dict, color_data:dict = None, color_bin_n=1):
    # x_data, y_data, color_data: Dict structs which contain the name of the data and the data itself

    n_bin = color_bin_n
    cmap = cm.get_cmap('cool',n_bin)
    plt.sca(ax)
    xlabel = list(xdata.keys())[0]
    ylabel = list(ydata.keys())[0]
    if color_data is None:
        plt.scatter(xdata[xlabel], ydata[ylabel], s=12)
    else:
        color_label = list(color_data.keys())[0]
        max_c_value = np.nanmax(color_data[color_label])
        min_c_value = np.nanmin(color_data[color_label])
        if max_c_value>min_c_value:
            bin_labels = np.arange(min_c_value, max_c_value-1e-10, (max_c_value-min_c_value)/n_bin)+(max_c_value-min_c_value)/n_bin/2
        else:
            bin_labels = np.ones(n_bin)
        category_result = np.array(pd.cut(color_data[color_label], n_bin, labels = bin_labels, ordered=False)) 

        plt.scatter(xdata[xlabel], ydata[ylabel], s=12, c=category_result, cmap=cmap, vmax = max_c_value, vmin = min_c_value)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(color_label, rotation=270)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Some statistics:
    (rho, p) = stats.spearmanr(xdata[xlabel], ydata[ylabel])
    plt.title(f"rho_all = {rho:.3f}, p = {p:.3f}")

def scatter_plot_color(x, y, z, x_label, y_label, z_label, title, ax=None, square=True, dot_size=12, vmin=None, vmax=None):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    if vmin is None or vmax is None:
        vmax = np.ceil(np.nanmax(z)*100)/100
        vmin = np.floor(np.nanmin(z)*100)/100 #0.0 
    cmap = plt.get_cmap('viridis')
    custom_cmap = create_custom_colormap(cmap, vmin, vmax)
    
    h = plt.scatter(x, y, c=z, cmap=custom_cmap, vmin=vmin, vmax=vmax, s=dot_size)
    # label the colorbar:
    cbar = plt.colorbar(h, ax=ax)
    cbar.set_label(z_label, rotation=270, labelpad=20)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if square:
        ax.set_aspect('equal', adjustable='box')
    return ax


def heatmap_plot(x, y, z, x_label, y_label, z_label, title, step = 0.01, ax=None, square=True, cmap='viridis', vmin=None, vmax=None, rescale_norm = True):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    if vmin is None or vmax is None:
        vmax = np.ceil(np.nanmax(z)*100)/100
        vmin = np.floor(np.nanmin(z)*100)/100
    xx, yy = np.meshgrid(x, y)
    if rescale_norm:
        norm = Rescaled_Norm(vmax=vmax, vmin = vmin)
    else:
        norm = None
    h = ax.contourf(xx, yy, z, cmap=cmap, vmin=vmin, vmax=vmax, levels = np.arange(vmin, vmax+0.001, step), norm=norm)
    # label the colorbar:
    cbar = plt.colorbar(h, ax=ax, ticks=np.arange(vmin, vmax+0.001, 0.1))
    cbar.set_label(z_label, rotation=270, labelpad=20)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if square:
        ax.set_aspect('equal', adjustable='box')
    return ax


def index_analysis_v4(W_12, W_21T, patterns_L1, patterns_L2, perc_active_L1=None, neuron_base_state = -1, makeplot = True, ax=None, bins = 10, color = 'r'):
    # compare the sign of W12 with the closest target patterns.
    # check if patterns is numpy array if not convert it to numpy array
    if isinstance(patterns_L1, torch.Tensor):
        patterns_L1 = patterns_L1.detach().numpy()
    if isinstance(patterns_L2, torch.Tensor):
        patterns_L2 = patterns_L2.detach().numpy()
    if isinstance(W_12, torch.Tensor):
        W_12 = W_12.detach().numpy()
    if isinstance(W_21T, torch.Tensor):
        W_21T = W_21T.detach().numpy()
    
    n_pattern = patterns_L1.shape[0]


    digitized_neuron_weight = np.sign(W_12).T
    digitized_neuron_weight_2 = np.sign(W_21T).T
    digitized_pattern_L1 = np.sign(patterns_L1-0.5)

    index_score_by_neuron = np.array([])
    for i in range(n_pattern):
        activated_neuron_L2 = patterns_L2[i]>0.5
        memory_L1 = digitized_pattern_L1[i]
        if np.sum(activated_neuron_L2)>0:
            activated_neuron_L2_indices = np.where(activated_neuron_L2)[0]
            for index in activated_neuron_L2_indices:
                sim_within_by_neuron = (calc_similarity_score(digitized_neuron_weight[index,:], memory_L1) + calc_similarity_score(digitized_neuron_weight_2[index,:], memory_L1))/2
                index_score_by_neuron = np.append(index_score_by_neuron, sim_within_by_neuron)


    if makeplot:
        # plot the distribution of the distance
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        plt.sca(ax)

        plt.hist(index_score_by_neuron, alpha=0.5,color=color, label='index score V3', bins = bins, density=True)
        plt.axvline(index_score_by_neuron.mean(), color=color, linestyle='dashed', linewidth=1)
        plt.xlabel('index score')

        plt.legend(loc='upper left')
    
    

    return index_score_by_neuron.mean().item()


def compare_rho_permutation_test(X1:np.ndarray, Y1:np.ndarray, X2:np.ndarray, Y2:np.ndarray, nperm = 5000):
    # input are two pairs of data, and we compare their correlation 
    # fix the # permutations

    size1 = X1.shape[0]
    size2 = X2.shape[0]

    # set a void vector for the dif of correl.
    corr_diff = np.zeros(nperm)

    # now start permuting
    for i in range(nperm):
        # sample an index
        if size1==size2:
            idx1 = random.choice(size1, size1, replace=True)
            idx2 = idx1
        else:
            idx1 = random.choice(size1, size1, replace=True)
            idx2 = random.choice(size2, size2, replace=True)

        # calculate the permuted correlation in the first condition
        (corr1, p) = stats.spearmanr(X1[idx1], Y1[idx1])

        # calculate the permuted correlation in the second condition
        (corr2, p) = stats.spearmanr(X2[idx2], Y2[idx2])

        # store the dif. of correlations
        corr_diff[i] = corr1-corr2

    
    # compute the Monte Carlo approximation of the permutation p-value
    if np.any((corr_diff>0) | (corr_diff<0)):
        p = 2*min(np.mean(corr_diff>0), np.mean(corr_diff<0))
    else:
        p = np.nan
    return p

