import numpy as np
from numpy import random
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import scipy
import scipy.stats as stats
from random_connection_network import random_connect_network
from numerical_random_network_parallel import calculate_numerical_cacpacity
import utils_basic as utils
import utils_numerical as utils_num
import copy
import pickle
import warnings
import matplotlib
import argparse
import sys
import time
# matplotlib.use('TkAgg')

import multiprocessing as mp

def individual_input_analysis(network, patterns):
    J = network.weight
    mask = network.mask
    bias = network.b
    n_neuron = network.n_neuron
    n_pattern = patterns.shape[0]

    total_input_matrix = torch.zeros((n_neuron, n_pattern))
    ratio_individual_input_matrix = torch.zeros((n_neuron, n_pattern))
    sign_individual_input_matrix = torch.zeros((n_neuron, n_pattern))
    for mm in range(n_pattern):
        pattern = patterns[mm]
        for ii in range(n_neuron):
            # calculate the mean of individual input:
            individual_inputs = J[ii,:]*pattern
            individual_inputs = individual_inputs[mask[ii,:] == 1]
            total_input_matrix[ii,mm] = torch.sum(individual_inputs)
            ratio_individual_input_matrix[ii,mm] = torch.sum(individual_inputs > 0)/torch.sum(mask[ii,:] == 1)
            sign_individual_input_matrix[ii,mm] =  pattern[ii]   #torch.sign(torch.sum(individual_inputs) + bias[ii])  

    mean_total_input = torch.mean(total_input_matrix, dim=1)
    ratio_individual_input = torch.mean(ratio_individual_input_matrix, dim=1)
    mean_sign = torch.mean(sign_individual_input_matrix, dim=1)
    return mean_total_input, ratio_individual_input, mean_sign
    
def scatter_plot_color(x, y, z, x_label, y_label, z_label, title, ax=None, square=True, size = 8, vmax=None, vmin=None):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    if vmax is None or vmin is None:
        vmax = np.ceil(np.nanmax(z)*100)/100
        vmin = np.floor(np.nanmin(z)*100)/100 #0.0 
    cmap = plt.get_cmap('viridis')
    custom_cmap = utils_num.create_custom_colormap(cmap, vmin, vmax)
    
    h = plt.scatter(x, y, c=z, cmap=custom_cmap, vmin=vmin, vmax=vmax, s=size) 
    # label the colorbar:
    cbar = plt.colorbar(h, ax=ax)
    cbar.set_label(z_label, rotation=270, labelpad=20)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if square:
        ax.set_aspect('equal', adjustable='box')
    return ax

if __name__ == '__main__':

    sys.argv += "--n_neuron 500 --n_repeat 10 --pattern_act_mean 0.25  --ratio_conn_mean 0.5 --change_in_degree\
        --heter_type both_uncorr --ratio_conn_std 0.10 --pattern_act_std 0.10 --seed 15".split()

    parser = argparse.ArgumentParser(description="Memorize correlated concepts with one layer network")
    parser.add_argument('--method', type=str, default='svm', help='training method', choices=['PLA', 'svm'])
    parser.add_argument('--n_neuron', type=int, default=50, help='number of neurons')
    parser.add_argument('--n_repeat', type=int, default=10, help='number of repeats')
    parser.add_argument('--kappa', type=float, default=0, help='kappa for the SVM')
    parser.add_argument('--use_reg', action='store_true', help='whether to use kappa')
    parser.add_argument('--weight_std_init', type=float, default=0.1, help='initial weight std')
    parser.add_argument('--neuron_base_state', type=int, default=-1, help='neuron base state')
    parser.add_argument('--W_symmetric', action='store_true', help='whether the weights are symmetric')
    parser.add_argument('--pattern_act_mean', type=float, default=0.25, help='mean of the neuron activation')
    parser.add_argument('--pattern_act_std', type=float, default=0.0, help='std of the neuron activation')
    parser.add_argument('--ratio_conn_mean', type=float, default=0.5, help='mean of the connection ratio')
    parser.add_argument('--ratio_conn_std', type=float, default=0.0, help='std of the connection ratio')
    parser.add_argument('--change_in_degree', action='store_true', help='change the in degree of the neurons')
    parser.add_argument('--change_out_degree', action='store_true', help='change the out degree of the neurons')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epsilon', type=float, default=0.01, help='error threshold')
    parser.add_argument('--training_max_iter', type=int, default=10000, help='max number of iterations for training')
    parser.add_argument('--lr_PLA', type=float, default=0.01, help='learning rate for PLA')
    # parser.add_argument('--use_coding_level_heter', action='store_true', help='whether to use heterogeneous coding level, default is heterogeneous network topology')
    parser.add_argument('--heter_type', type=str, default='ratio_conn_std', choices=['both_positive_corr', 'both_negative_corr', 'both_uncorr'], help='heterogeneity type')
    args = parser.parse_args()

    pars = vars(args)

    pars['W_notsymmetric'] = not pars['W_symmetric']

    method = pars['method']
    n_neuron = pars['n_neuron']
    n_repeat = pars['n_repeat']
    kappa = pars['kappa']
    use_reg = pars['use_reg']
    weight_std_init = pars['weight_std_init']
    neuron_base_state = pars['neuron_base_state']
    W_symmetric = pars['W_symmetric']

    pattern_act_mean = pars['pattern_act_mean']
    pattern_act_std = pars['pattern_act_std']
    ratio_conn_mean = pars['ratio_conn_mean']
    ratio_conn_std = pars['ratio_conn_std']
    heter_type = pars['heter_type']
    epsilon = pars['epsilon']

    seed = pars['seed']

    # set the seed
    random.seed(seed)
    torch.manual_seed(seed)
    t_start = time.time()


    heter_type_set = ['both_positive_corr', 'both_uncorr','both_negative_corr']
    plot_index =  [0, 1, 2]
    color_set = ['red', 'olive', 'blue']
    label_set = ['positive correlation', 'uncorrelated', 'negative correlation']


    # make plot:
    fig, axes = plt.subplots(3,5, figsize = [18,10])
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.4, hspace=0.4)

    threshold_set = np.array([])
    ratio_individual_input_set = np.array([])
    mean_total_input_set = np.array([])
    coding_level_set = np.array([])

    for i, heter_type in enumerate(heter_type_set):
        # make the mask and coding level for the network
        neuron_act_prop = (torch.rand((n_neuron,))-0.5)*(pattern_act_std*np.sqrt(12))+pattern_act_mean

        if pars["change_in_degree"]:
            connection_prop_in = (torch.rand((n_neuron,))-0.5)*(ratio_conn_std*np.sqrt(12))+ratio_conn_mean
        else:
            connection_prop_in = torch.ones(n_neuron)*ratio_conn_mean
        if pars["change_out_degree"]:
            connection_prop_out = (torch.rand((n_neuron,))-0.5)*(ratio_conn_std*np.sqrt(12))+ratio_conn_mean
        else:
            connection_prop_out = torch.ones(n_neuron)*ratio_conn_mean

        if heter_type == 'both_positive_corr':
            # rearrange the neuron_act_prop such that larger neuron_act_prop is associated with larger connection_prop_in
            connection_prop_in_in_order, sorted_ind = connection_prop_in.sort()
            neuron_act_prop_in_order = neuron_act_prop.sort()[0]
            recover_ind = torch.argsort(sorted_ind)
            neuron_act_prop = neuron_act_prop_in_order[recover_ind]
        elif heter_type == 'both_negative_corr':
            # rearrange the neuron_act_prop such that larger neuron_act_prop is associated with smaller connection_prop_in
            connection_prop_in_in_order, sorted_ind = connection_prop_in.sort()
            neuron_act_prop_in_order = neuron_act_prop.sort()[0]
            recover_ind = torch.argsort(sorted_ind)
            neuron_act_prop = torch.flip(neuron_act_prop_in_order, dims=[0])[recover_ind]
        elif heter_type == 'both_uncorr':
            # do nothing
            pass

        mask_prob = (connection_prop_in[:, None] * connection_prop_out[None,:])/ratio_conn_mean
        mask_prob = torch.clip(mask_prob, 0, 1)
        mask = torch.bernoulli(mask_prob)
        mask.fill_diagonal_(0)

        # make the network:
        network = random_connect_network(n_neuron, W_symmetric=W_symmetric,connection_prop=None, weight_std_init=weight_std_init, mask=mask, neuron_base_state=neuron_base_state)

        # find the capacity of network:
        cap = calculate_numerical_cacpacity(pars, n_neuron*7, 2, epsilon=epsilon, neuron_act_prop=neuron_act_prop, network=network)
        print("numerical capacity: ", cap)

        network.reinitiate()
        # train the network to store patterns:
        patterns = utils.make_pattern(cap, n_neuron, perc_active=neuron_act_prop)
        if method == 'PLA':
            raise NotImplementedError("PLA method is outdated, please use SVM method")
        elif method == 'svm':
            success, stored_patterns_all = network.train_svm(patterns, use_reg = use_reg)

        # calculate the error
        error = 1-utils.network_score(network.weight, network.b, patterns, kappa = kappa)
        print("error: ", error)

        mean_total_input, ratio_individual_input, mean_sign = individual_input_analysis(network, patterns)
        empirical_coding_level = (mean_sign+1)/2 #neuron_act_prop #(mean_sign+1)/2 # neuron_act_prop
        threshold = -network.b.detach().numpy()
        pos_weight_ratio = torch.sum(network.weight > 0, dim=1)/torch.sum(mask, dim=1)
        in_degree = torch.sum(mask, dim=1)/n_neuron
        vmax = 0.75
        vmin = 0.25

        plt.sca(axes[0,plot_index[i]])
        scatter_plot_color(empirical_coding_level, ratio_individual_input, in_degree, "Neural coding level", "Percentage of positive input", "In-degree", f"{label_set[i]}", ax=axes[0,plot_index[i]], square=False, vmax=vmax, vmin=vmin)
        # plt.scatter(empirical_coding_level, ratio_individual_input, label=label_set[i],alpha=0.5, s=8)
        # use least square to fit the line:
        slope, intercept, r_value, p_value, std_err = stats.linregress(empirical_coding_level.numpy(), ratio_individual_input.numpy())
        plt.plot(empirical_coding_level, slope*empirical_coding_level + intercept, color= color_set[i])
        plt.xlabel("Neural coding level")
        plt.ylabel("Percentage of positive input")
        # print the correlation coefficient and p value on the last subplot
        plt.sca(axes[0,3])
        plt.text(0.5, 0.5-i*0.1, label_set[i] + f" r = {r_value:.7f}, p = {p_value:.7f}", color=color_set[i], fontsize=6)
        plt.xlim(0.5,1)
        
        plt.sca(axes[1,plot_index[i]])
        scatter_plot_color(empirical_coding_level, mean_total_input, in_degree, "Neural coding level", "Mean of the total input", "In-degree", "", ax=axes[1,plot_index[i]], square=False, vmax=vmax, vmin=vmin)
        # plt.scatter(empirical_coding_level, mean_total_input, label=label_set[i], color=color_set[i], alpha=0.5, s=8)
        # use least square to fit the line:
        slope, intercept, r_value, p_value, std_err = stats.linregress(empirical_coding_level.numpy(), mean_total_input.numpy())
        plt.plot(empirical_coding_level, slope*empirical_coding_level + intercept, color= color_set[i])
        plt.xlabel("Neural coding level")
        plt.ylabel("Total input")
        plt.sca(axes[1,3])
        plt.text(0.5, 0.5-i*0.1, label_set[i] + f" r = {r_value:.7f}, p = {p_value:.7f}", color=color_set[i], fontsize=6)
        plt.xlim(0.5,1)

        plt.sca(axes[2,plot_index[i]])
        scatter_plot_color(empirical_coding_level, threshold, in_degree, "Neural coding level", "Activation Threshold", "In-degree", "", ax=axes[2,plot_index[i]], square=False, vmax=vmax, vmin=vmin)
        # plt.scatter(empirical_coding_level, threshold, label=label_set[i], color=color_set[i], alpha=0.5, s=8)
        # use least square to fit the line:
        slope, intercept, r_value, p_value, std_err = stats.linregress(empirical_coding_level.numpy(), threshold)
        plt.plot(empirical_coding_level, slope*empirical_coding_level + intercept, color= color_set[i])
        plt.xlabel("Neural coding level")
        plt.ylabel("Activation Threshold")
        plt.sca(axes[2,3])
        plt.text(0.5, 0.5-i*0.1, label_set[i] + f" r = {r_value:.7f}, p = {p_value:.7f}", color=color_set[i], fontsize=6)
        plt.xlim(0.5,1)


        # plt.sca(axes[1,0])
        # plt.scatter(empirical_coding_level, mean_sign, label=label_set[i], color=color_set[i], alpha=0.5)
        # # use least square to fit the line:
        # slope, intercept = np.polyfit(empirical_coding_level.numpy(), mean_sign.numpy(), 1)
        # plt.plot(empirical_coding_level, slope*empirical_coding_level + intercept, color= color_set[i])
        # plt.xlabel("Neural coding level")
        # plt.ylabel("Mean of the activity of neurons (control analysis)")
        threshold_set = np.append(threshold_set, threshold)
        ratio_individual_input_set = np.append(ratio_individual_input_set, ratio_individual_input)
        coding_level_set = np.append(coding_level_set, empirical_coding_level)
        mean_total_input_set = np.append(mean_total_input_set, mean_total_input)

    # make the axis the same range:
    threshold_range = [threshold_set.min()-2, threshold_set.max()+2]
    ratio_individual_input_range = [ratio_individual_input_set.min()-0.02, ratio_individual_input_set.max()+0.02]
    total_input_range = [mean_total_input_set.min()-2, mean_total_input_set.max()+2]
    coding_level_range = [coding_level_set.min()-0.02, coding_level_set.max()+0.02]
    for i in plot_index:
        plt.sca(axes[0,i])
        plt.xlim(coding_level_range)
        plt.ylim(ratio_individual_input_range)
        plt.sca(axes[1,i])
        plt.xlim(coding_level_range)
        plt.ylim(total_input_range)
        plt.sca(axes[2,i])
        plt.xlim(coding_level_range)
        plt.ylim(threshold_range)

    # plot the legend
    plt.sca(axes[0,0])
    plt.legend()
    plt.sca(axes[0,1])
    plt.legend()
    plt.sca(axes[0,2])
    plt.legend()

    utils.print_parameters_on_plot(axes[2,4], pars)

    plt.show(block=False)
    plt.savefig('results/individual_input_analysis'+utils.make_name_with_time()+'.png')
    plt.savefig('results/individual_input_analysis'+utils.make_name_with_time()+'.pdf')

    t_end = time.time()
    print(f"Computation finished, time used: {t_end-t_start} seconds")

    print("done")







