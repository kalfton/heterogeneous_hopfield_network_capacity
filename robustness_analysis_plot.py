import numpy as np
from numpy import random
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import matplotlib
import scipy
from random_connection_network import random_connect_network
import utils_basic as utils
import utils_numerical as utils_num
import copy
# import utils_numerical as utils_num
import pickle
import warnings
import time
import argparse



n_neuron_set = [50, 150, 500]
n_neuron_L2_set = [150, 450, 1500]

# color_map_range = [0.5, 0.7] # the range of the color map for the scatter plot.



# load the capacity values from theoretical analysis:
with open('results/theoretical_two_layer_capacity_L1_150_L2_450.pkl', 'rb') as f:
    capacities_theoretical, information_capacity_theoretical, _, _ = pickle.load(f)

fig, axes = plt.subplots(3,3, figsize = [19,10])
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.4, hspace=0.4)

for i, n_neuron in enumerate(n_neuron_set):
    n_neuron_L2 = n_neuron_L2_set[i]
    print(f"n_neuron = {n_neuron}, n_neuron_L2 = {n_neuron_L2}")
    # load the results:
    with open(f'results/robustness_score_two_layer_V7_ablation=1 k = 0.0, n_layer1 = {n_neuron}, n_layer_2 = {n_neuron_L2}.pkl', 'rb') as f:
        robustness_score, hipp_index_score, perc_act_L1_set, perc_act_L2_set, pars = pickle.load(f)

    with open(f'results/robustness_score_two_layer_ideal_V7_ablation=1 k = 0.0, n_layer1 = {n_neuron}, n_layer_2 = {n_neuron_L2}.pkl', 'rb') as f:
        robustness_score_ideal, hipp_index_score_ideal, capacities_ideal, perc_act_L1_set, pars = pickle.load(f)

    capacities_ideal = capacities_ideal/(pars["n_neuron"]+pars["n_neuron_L2"])
    information_capacities_ideal = np.zeros_like(capacities_ideal)
    for j in range(len(perc_act_L1_set)):
        information_capacities_ideal[j] = capacities_ideal[j]*utils.information_entropy(perc_act_L1_set[j])*pars["n_neuron"]/(pars["n_neuron"]+pars["n_neuron_L2"])

    # plot the scatter of the lower branch:
    midpoint = int(len(perc_act_L2_set)/2)

    Info_times_memory_theoretical = information_capacity_theoretical*capacities_theoretical
    normalization_factor = np.max(Info_times_memory_theoretical)
    Info_times_memory_theoretical = Info_times_memory_theoretical/normalization_factor
    Info_times_memory_ideal = information_capacities_ideal*capacities_ideal/normalization_factor

    fig, axes = plt.subplots(3,3, figsize = [19,10])

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.4, hspace=0.4)

    utils_num.heatmap_plot(perc_act_L1_set, perc_act_L2_set, robustness_score, \
                "Activation probability in layer 1", "Activation probabiltiy in layer 2", "Robustness score", "The robustness score of the two-layer network", ax=axes[0,0], vmax = 1.0, vmin = 0.5, cmap='inferno', rescale_norm=False)

    utils_num.heatmap_plot(perc_act_L1_set, perc_act_L2_set, hipp_index_score,\
                "Activation probability in layer 1", "Activation probabiltiy in layer 2", "Hippo index score", "Hippo index score", step = 0.005, ax=axes[0,1])

    # plot the scatter of the lower branch:
    midpoint = int(len(perc_act_L2_set)/2)
    
    utils_num.scatter_plot_color(np.concat((capacities_theoretical[0:midpoint,:].flatten(), capacities_ideal.flatten())), np.concat((robustness_score[0:midpoint,:].flatten(), robustness_score_ideal.flatten())), np.concat((hipp_index_score[0:midpoint,:].flatten(), hipp_index_score_ideal.flatten())), \
                        "Memory capacity", "Robustness score", "Hippo index score", "The relationship between the robustness score and the hippo index score with ideal hippocampal index network", ax=axes[2,0], square=False)

    utils_num.scatter_plot_color(np.concat((information_capacity_theoretical[0:midpoint,:].flatten(), information_capacities_ideal.flatten())), np.concat((robustness_score[0:midpoint,:].flatten(), robustness_score_ideal.flatten())), np.concat((hipp_index_score[0:midpoint,:].flatten(), hipp_index_score_ideal.flatten())), \
                        "Information capacity", "Robustness score", "Hippo index score", "The relationship between the robustness score and the hippo index score with ideal hippocampal index network", ax=axes[2,1], square=False)
    Info_times_memory_theoretical = information_capacity_theoretical*capacities_theoretical
    normalization_factor = np.max(Info_times_memory_theoretical)

    Info_times_memory_theoretical = Info_times_memory_theoretical/normalization_factor
    Info_times_memory_ideal = information_capacities_ideal*capacities_ideal/normalization_factor

    utils_num.scatter_plot_color(Info_times_memory_theoretical[0:midpoint,:].flatten(), robustness_score[0:midpoint,:].flatten(), hipp_index_score[0:midpoint,:].flatten(), \
                        "Information times memory", "Robustness score", "Hippo index score", "Info times memory theoretical", ax=axes[2,2], square=False)
    
    cbar = plt.colorbar(axes[2,2].collections[0], ax=axes[2,2])
    cbar.set_ticks(np.arange(0.5, 0.61, 0.02))
    # scatter_plot_color(Info_times_memory_ideal, robustness_score_ideal, hipp_index_score_ideal, \
    #                         "Information times memory", "Robustness score", "Hippo index score", "", ax=axes[2,2], square=False)
    plt.sca(axes[2,2])
    plt.scatter(Info_times_memory_ideal, robustness_score_ideal, c=hipp_index_score_ideal, cmap='cool', vmin=0, vmax=1, s=12)

    axes[2,2].set_aspect(2.5)
    utils.print_parameters_on_plot(axes[0,2], pars)



    plt.show(block=False)
    # plt.savefig(f'results/robustness_score_two_layer_V7_ablation=001 k = 0.0, n_layer1 = {n_neuron}, n_layer_2 = {n_neuron_L2}'+utils_num.make_name_with_time()+'.png')
    plt.savefig(f'results/robustness_score_two_layer_V7_ablation=001 k = 0.0, n_layer1 = {n_neuron}, n_layer_2 = {n_neuron_L2}'+utils_num.make_name_with_time()+'.pdf')

print("end")
