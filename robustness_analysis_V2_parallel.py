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

import multiprocessing as mp


def calc_robustness_score(network: random_connect_network, stored_patterns, ablation_unit_matrix, N_feature_layer, print_debug = False):
    """
    Compute the robustness of the two-layer network with respect to the removal of nodes.
    return a robustness score.
    """
    retrieval_pattern = network.forward(stored_patterns*ablation_unit_matrix,  n_step = 1, dt_train = 1)
    retrieval_pattern_feature = (retrieval_pattern[:,:N_feature_layer]).numpy()
    stored_patterns_feature = (stored_patterns[:,:N_feature_layer]).numpy()
    # N_act = torch.sum(stored_patterns_feature>0)
    # N_act_to_act = torch.sum((stored_patterns_feature>0) & (retrieval_pattern_feature>0))
    # N_inact = torch.sum(stored_patterns_feature<0)
    # N_inact_to_inact = torch.sum((stored_patterns_feature<0) & (retrieval_pattern_feature<0))

    # robustness = ((N_act_to_act/N_act) + (N_inact_to_inact/N_inact))/2
    robustness_by_pattern = np.zeros(stored_patterns_feature.shape[0])
    for mu in range(stored_patterns_feature.shape[0]): # for each pattern, we can calculate the similarity score, then average it over all patterns.
        robustness_by_pattern[mu] = utils.calc_similarity_score(stored_patterns_feature[mu], retrieval_pattern_feature[mu])
    robustness = np.mean(robustness_by_pattern) # average over all patterns, this is the final robustness score.

    # robustness = ((N_act_to_act + N_inact_to_inact)/(N_act + N_inact))
    # # for debug:
    # if print_debug:
    #     print("N_act = %d, N_act_to_act = %d, N_inact = %d, N_inact_to_inact = %d" % (N_act, N_act_to_act, N_inact, N_inact_to_inact))
    #     print("N_act_to_act/N_act = %f, N_inact_to_inact/N_inact = %f" % (N_act_to_act/N_act, N_inact_to_inact/N_inact))
    #     print("(N_act_to_act + N_inact_to_inact)/(N_act + N_inact) = %f" % ((N_act_to_act + N_inact_to_inact)/(N_act + N_inact)))
    #     n_pattern = stored_patterns.shape[0]
    #     print("empirical activation probability = %f" % (torch.sum(stored_patterns_feature>0)/(N_feature_layer*n_pattern)))
    
    return robustness.item()

def network_robustness(network: random_connect_network, stored_patterns, N_feature_layer, N_neuron, N_removal=1):
    """
    Compute the robustness of the two-layer network with respect to the removal of nodes.
    return a robustness score.
    """ 
    # Copy the network
    network= copy.deepcopy(network)

    N_neuron = network.n_neuron
    N_memory_layer = N_neuron - N_feature_layer

    repeatition = N_memory_layer

    # Compute the number of nodes to remove
    num_nodes_to_remove = int(N_removal)

    # # Calculate the initial voliation of stability condition:
    # initial_violation = 1- utils.network_score(network.weight, network.b, stored_patterns)

    # Calculate the stability condition after removing one node
    repeatition = N_memory_layer

    # Remove nodes
    robustness_all = []
    for i in range(repeatition):


        ablation_units_matrix = torch.ones_like(stored_patterns) # the ablation matrix represents the units that should be removed for each pattern, 0 means the unit is removed, 1 means the unit is kept. 
        # ablation_nodes = N_feature_layer + random.choice(range(N_memory_layer), num_nodes_to_remove, replace=False)
        # ablation_units_matrix[:, ablation_nodes] = 0
        for j in range(stored_patterns.shape[0]):
            # only ablating the neurons that are activated in the j-th pattern:
            unit_index = torch.arange(N_neuron)
            active_units_memory_layer = torch.where(torch.logical_and(stored_patterns[j]>0, unit_index>=N_feature_layer))[0]
            if len(active_units_memory_layer)>num_nodes_to_remove:
                ablation_units = random.choice(active_units_memory_layer, num_nodes_to_remove, replace=False)
                ablation_units_matrix[j, ablation_units] = 0
            else:
                ablation_units_matrix[j, active_units_memory_layer] = 0

        robustness = calc_robustness_score(network, stored_patterns, ablation_units_matrix, N_feature_layer)
        robustness_all.append(robustness)
    robustness = torch.mean(torch.tensor(robustness_all))

    robustness = torch.mean(torch.tensor(robustness_all))


    return robustness.item(), torch.nan



def heatmap_plot(x, y, z, x_label, y_label, z_label, title, step = 0.01, ax=None, square=True):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    vmax = np.ceil(np.nanmax(z)*100)/100
    vmin = np.floor(np.nanmin(z)*100)/100 #0.0 
    xx, yy = np.meshgrid(x, y)
    h = ax.contourf(xx, yy, z, cmap='viridis', vmin=vmin, vmax=vmax, levels = np.arange(vmin, vmax+0.001, step), norm=utils_num.Rescaled_Norm(vmax=vmax, vmin = vmin))
    # label the colorbar:
    cbar = plt.colorbar(h, ax=ax)
    cbar.set_label(z_label, rotation=270, labelpad=20)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if square:
        ax.set_aspect('equal', adjustable='box')
    return ax

def calculate_robustness_and_index_score(capacity, perc_act_L1, perc_act_L2, pars):
    # parameters:
    n_neuron = pars["n_neuron"]
    n_neuron_L2 = pars["n_neuron_L2"]
    neuron_base_state = pars["neuron_base_state"]
    network_type = pars["network_type"]
    W_symmetric = not pars["W_notsymmetric"]


    weight_std_init = pars["weight_std_init"]
    method = pars["method"]
    lr_PLA = pars["lr_PLA"]
    training_max_iter = pars["training_max_iter"]
    N_removal = pars["N_removal"]

    n_pattern = int(np.floor(capacity))
    if n_pattern <=2:
        robustness_score = np.nan
        hipp_index_score = np.nan
        return robustness_score, hipp_index_score
    # make the patterns:
    patterns_L1 = utils.make_pattern(n_pattern, n_neuron, perc_active=perc_act_L1)
    patterns_L2 = utils.make_pattern(n_pattern, n_neuron_L2, perc_active=perc_act_L2)

    # initialize the network:
    if network_type.lower() == 'RBM'.lower():
        mask = torch.ones((n_neuron+n_neuron_L2, n_neuron+n_neuron_L2))
        mask[:n_neuron, :n_neuron] = 0
        mask[n_neuron:, n_neuron:] = 0
        mask.fill_diagonal_(0)
    else:
        raise ValueError("network_type should be 'RBM', 'Hybrid', or 'Full'.")
    
    network = random_connect_network(n_neuron+n_neuron_L2, W_symmetric=W_symmetric,connection_prop=None, weight_std_init=weight_std_init, mask=mask, neuron_base_state=neuron_base_state)
    patterns_all = torch.cat((patterns_L1, patterns_L2), dim=1)
    if method.lower() == 'PLA'.lower():
        success, stored_patterns_all = network.train_PLA(patterns_all, training_max_iter=training_max_iter, lr=lr_PLA)
    elif method.lower() == 'svm'.lower():
        if pars["use_kappa"]:
            success, stored_patterns_all = network.train_svm(patterns_all, use_reg = True, kappa = pars["kappa"])
        else:
            success, stored_patterns_all = network.train_svm(patterns_all, use_reg = False)


    # index theory analysis:
    W_12_trained = network.weight.clone().numpy()[:n_neuron, n_neuron:]
    W_21_trained = network.weight.clone().numpy()[n_neuron:, :n_neuron]

    (robustness_score,_)= network_robustness(network, patterns_all, n_neuron, n_neuron_L2, N_removal=N_removal)
    dists = utils_num.index_analysis_v4(W_12_trained, W_21_trained.T, patterns_L1=patterns_L1,patterns_L2=patterns_L2, perc_active_L1=perc_act_L1, neuron_base_state = neuron_base_state, makeplot=False)
    hipp_index_score = dists

    return robustness_score, hipp_index_score


if __name__ == '__main__':

    # load the parameters for numerical simulation:
    with open('results/numerical_two_layer_temp_k = 0, n_layer1 = 150, n_layer_2 = 450_cluster.pkl', 'rb') as f:
        capacities_num, information_capacity_num, perc_act_L1_set, perc_act_L2_set, pars = pickle.load(f)

    capacities_num = capacities_num/(pars["n_neuron"]+pars["n_neuron_L2"])
    information_capacity_num = information_capacity_num/(pars["n_neuron"]+pars["n_neuron_L2"])**2

    # load the capacity values from theoretical analysis:
    with open('results/theoretical_two_layer_capacity_L1_150_L2_450.pkl', 'rb') as f:
        capacities_theoretical, information_capacity_theoretical, _, _ = pickle.load(f)

    # parameters:
    n_neuron = pars["n_neuron"]
    n_neuron_L2 = pars["n_neuron_L2"]
    neuron_base_state = pars["neuron_base_state"]
    network_type = pars["network_type"]
    W_symmetric = not pars["W_notsymmetric"]


    weight_std_init = pars["weight_std_init"]
    method = pars["method"]
    lr_PLA = pars["lr_PLA"]
    training_max_iter = pars["training_max_iter"]
    n_process = min(mp.cpu_count(), 15)


    # rescale the neuron number
    n_neuron = 50
    n_neuron_L2 = 150
    pars["n_neuron"] = n_neuron
    pars["n_neuron_L2"] = n_neuron_L2
    N_removal = 1 # 0.01 # 1/n_neuron_L2
    pars["N_removal"] = N_removal



    # capacities = (np.floor(capacities_theoretical*(n_neuron+n_neuron_L2))).astype(int)
    # information_capacity = information_capacity_theoretical*(n_neuron+n_neuron_L2)**2


    # robustness_score = np.zeros((len(perc_act_L2_set), len(perc_act_L1_set)))
    # hipp_index_score = np.zeros((len(perc_act_L2_set), len(perc_act_L1_set)))


    # with mp.Pool(n_process) as pool:
    #     pool_results_object = [[[] for i in range(len(perc_act_L2_set))] for j in range(len(perc_act_L1_set))]
    #     for i in range(len(perc_act_L2_set)):
    #         for j in range(len(perc_act_L1_set)):
    #             print("perc_act_L1 = %f, perc_act_L2 = %f, capacity = %f" % (perc_act_L1_set[j], perc_act_L2_set[i], capacities[i][j]))
    #             pool_results_object[i][j] = pool.apply_async(calculate_robustness_and_index_score, args=(capacities[i][j], perc_act_L1_set[j], perc_act_L2_set[i], pars))
    #             # # for debug:
    #             # robustness_score[i][j], hipp_index_score[i][j] = calculate_robustness_and_index_score(capacities[i][j], perc_act_L1_set[j], perc_act_L2_set[i], pars)
    #             # print("robustness_score[%d][%d] = %f, hipp_index_score[%d][%d] = %f" % (i, j, robustness_score[i][j], i, j, hipp_index_score[i][j]))
                
    #             # n_pattern = int(np.floor(capacities[i][j]))
    #             # if n_pattern <=2:
    #             #     robustness_score[i,j] = np.nan
    #             #     continue
    #             # print("n_pattern = %d" % n_pattern)
    #             # # make the patterns:
    #             # patterns_L1 = utils.make_pattern(n_pattern, n_neuron, perc_active=perc_act_L1_set[j], neuron_base_state=neuron_base_state)
    #             # patterns_L2 = utils.make_pattern(n_pattern, n_neuron_L2, perc_active=perc_act_L2_set[i], neuron_base_state=neuron_base_state)

    #             # # initialize the network:
    #             # if network_type.lower() == 'RBM'.lower():
    #             #     mask = torch.ones((n_neuron+n_neuron_L2, n_neuron+n_neuron_L2))
    #             #     mask[:n_neuron, :n_neuron] = 0
    #             #     mask[n_neuron:, n_neuron:] = 0
    #             #     mask.fill_diagonal_(0)
    #             # else:
    #             #     raise ValueError("network_type should be 'RBM', 'Hybrid', or 'Full'.")
                
    #             # network = random_connect_network(n_neuron+n_neuron_L2, W_symmetric=W_symmetric,connection_prop=None, weight_std_init=weight_std_init, mask=mask, neuron_base_state=neuron_base_state)
    #             # patterns_all = torch.cat((patterns_L1, patterns_L2), dim=1)
    #             # if method.lower() == 'PLA'.lower():
    #             #     success, stored_patterns_all = network.train_PLA(patterns_all, training_max_iter=training_max_iter, lr=lr_PLA)
    #             # elif method.lower() == 'svm'.lower():
    #             #     if pars["use_kappa"]:
    #             #         success, stored_patterns_all = network.train_svm(patterns_all, use_reg = True, kappa = pars["kappa"])
    #             #     else:
    #             #         success, stored_patterns_all = network.train_svm(patterns_all, use_reg = False)


    #             # # index theory analysis:
    #             # W_12_trained = network.weight.clone().numpy()[:n_neuron, n_neuron:]
    #             # W_21_trained = network.weight.clone().numpy()[n_neuron:, :n_neuron]

    #             # (robustness_score[i,j],_)= network_robustness(network, patterns_all, n_neuron, n_neuron_L2, N_removal=N_removal)
    #             # dists = utils_num.index_analysis_v4(W_12_trained, W_21_trained.T, patterns_L1=patterns_L1,patterns_L2=patterns_L2, perc_active_L1=perc_act_L1_set[j], neuron_base_state = neuron_base_state, makeplot=False)
    #             # hipp_index_score[i,j] = dists

    #     for i in range(len(perc_act_L2_set)):
    #         for j in range(len(perc_act_L1_set)):
    #             try:
    #                 robustness_score[i,j], hipp_index_score[i,j] = pool_results_object[i][j].get()
    #             except:
    #                 raise ValueError("Error in getting the results of pool_results_object[%d][%d]" % (i,j))
            
    #             print("perc_act_L1 = %f, perc_act_L2 = %f, capacity = %f, robustness_score = %f, hipp_index_score = %f" % (perc_act_L1_set[j], perc_act_L2_set[i], capacities[i][j], robustness_score[i,j], hipp_index_score[i,j]))


    # # make a network that use ideal hippo index theory to store memory and its robustness score:
    # mask = torch.ones((n_neuron+n_neuron_L2, n_neuron+n_neuron_L2))
    # mask[:n_neuron, :n_neuron] = 0
    # mask[n_neuron:, n_neuron:] = 0
    # mask.fill_diagonal_(0)

    # robustness_score_ideal = np.zeros(len(perc_act_L1_set))
    # hipp_index_score_ideal = np.zeros(len(perc_act_L1_set))
    # capacities_ideal = np.zeros(len(perc_act_L1_set))
    # for j in range(len(perc_act_L1_set)):
    #     n_pattern = n_neuron_L2
    #     patterns_L1 = utils.make_pattern(n_pattern, n_neuron, perc_active=perc_act_L1_set[j], neuron_base_state=neuron_base_state)
    #     # remove replicate patterns:
    #     patterns_L1 = torch.unique(patterns_L1, dim=0)
    #     if patterns_L1.shape[0] < n_pattern:
    #         print(f"n_pattern = {n_pattern}, but the number of unique patterns is {patterns_L1.shape[0]}, skip this case.")
    #         robustness_score_ideal[j] = np.nan
    #         hipp_index_score_ideal[j] = np.nan
    #         capacities_ideal[j] = np.nan
    #         continue
    #     patterns_L2 = torch.diag(torch.ones(n_neuron_L2))*2-1
    #     patterns_L2 = patterns_L2[:n_pattern,:]
    #     patterns_all = torch.cat((patterns_L1, patterns_L2), dim=1)
        

    #     network = random_connect_network(n_neuron+n_neuron_L2, W_symmetric=W_symmetric,connection_prop=None, weight_std_init=weight_std_init, mask=mask, neuron_base_state=neuron_base_state)
    #     cross_weight = torch.zeros((n_neuron, n_neuron_L2))
    #     for i in range(n_neuron_L2):
    #         cross_weight[:, i] = patterns_L1[i]
    #     network.weight = torch.zeros((n_neuron+n_neuron_L2, n_neuron+n_neuron_L2))
    #     network.weight[:n_neuron, n_neuron:] = cross_weight
    #     network.weight[n_neuron:, :n_neuron] = cross_weight.T
        
    #     bias_L1 = torch.zeros(n_neuron)
    #     for i in range(n_neuron_L2):
    #         bias_L1 = -(torch.max(torch.matmul(cross_weight, patterns_L2.T), dim=1)[0]-0.5)
    #     bias_L2 = -(torch.max(torch.matmul(cross_weight.T, patterns_L1.T), dim=1)[0]-0.5)
    #     network.b = torch.cat((bias_L1, bias_L2), dim=0)
    #     # check that the patterns are stored correctly:
    #     stored_patterns_all = network.forward(patterns_all, n_step = 1, dt_train = 1)
    #     if torch.sum(stored_patterns_all*patterns_all <0 ) > 1:
    #         print(f"The stored patterns are not correct! perc_act_L1 = {perc_act_L1_set[j]}")
    #     (robustness_score_ideal[j],_)= network_robustness(network, patterns_all, n_neuron, n_neuron_L2, N_removal=N_removal)

    #     W_12_trained = network.weight.clone().numpy()[:n_neuron, n_neuron:]
    #     W_21_trained = network.weight.clone().numpy()[n_neuron:, :n_neuron]
    #     hipp_index_score_ideal[j] = utils_num.index_analysis_v4(W_12_trained, W_21_trained.T, patterns_L1=patterns_L1,patterns_L2=patterns_L2, perc_active_L1=perc_act_L1_set[j], neuron_base_state = neuron_base_state, makeplot=False)

    #     capacities_ideal[j] = n_neuron_L2


    # # save the results:
    # with open(f'results/robustness_score_two_layer_V7_ablation=1 k = 0.0, n_layer1 = {n_neuron}, n_layer_2 = {n_neuron_L2}.pkl', 'wb') as f:
    #     pickle.dump((robustness_score, hipp_index_score, perc_act_L1_set, perc_act_L2_set, pars), f)

    # with open(f'results/robustness_score_two_layer_ideal_V7_ablation=1 k = 0.0, n_layer1 = {n_neuron}, n_layer_2 = {n_neuron_L2}.pkl', 'wb') as f:
    #     pickle.dump((robustness_score_ideal, hipp_index_score_ideal, capacities_ideal, perc_act_L1_set, pars), f)

    # load the results:
    with open(f'results/robustness_score_two_layer_V7_ablation=1 k = 0.0, n_layer1 = {n_neuron}, n_layer_2 = {n_neuron_L2}.pkl', 'rb') as f:
        robustness_score, hipp_index_score, perc_act_L1_set, perc_act_L2_set, pars = pickle.load(f)

    with open(f'results/robustness_score_two_layer_ideal_V7_ablation=1 k = 0.0, n_layer1 = {n_neuron}, n_layer_2 = {n_neuron_L2}.pkl', 'rb') as f:
        robustness_score_ideal, hipp_index_score_ideal, capacities_ideal, perc_act_L1_set, pars = pickle.load(f)

    capacities_ideal = capacities_ideal/(pars["n_neuron"]+pars["n_neuron_L2"])
    information_capacities_ideal = np.zeros_like(capacities_ideal)
    for j in range(len(perc_act_L1_set)):
        information_capacities_ideal[j] = capacities_ideal[j]*utils.information_entropy(perc_act_L1_set[j])*pars["n_neuron"]/(pars["n_neuron"]+pars["n_neuron_L2"])

    # # testing normalize the hippo index score:
    # hipp_index_score = hipp_index_score/ hipp_index_score_ideal
    # hipp_index_score_ideal = hipp_index_score_ideal/ hipp_index_score_ideal
    # plot the results:

    fig, axes = plt.subplots(3,3, figsize = [19,10])

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.4, hspace=0.4)

    heatmap_plot(perc_act_L1_set, perc_act_L2_set, robustness_score, \
                "Activation probability in layer 1", "Activation probabiltiy in layer 2", "Robustness score", "The robustness score of the two-layer network", ax=axes[0,0])

    heatmap_plot(perc_act_L1_set, perc_act_L2_set, hipp_index_score,\
                "Activation probability in layer 1", "Activation probabiltiy in layer 2", "Hippo index score", "Hippo index score", step = 0.005, ax=axes[0,1])
    # set the color bar ticks:
    cbar = plt.colorbar(axes[0,1].collections[0], ax=axes[0,1])
    cbar.set_ticks(np.arange(0.5, 0.61, 0.02))

    heatmap_plot(perc_act_L1_set, perc_act_L2_set, information_capacity_num, \
                "Activation probability in layer 1", "Activation probabiltiy in layer 2", "Information capacity", "The information capacity of the two-layer network", ax=axes[1,0])

    heatmap_plot(perc_act_L1_set, perc_act_L2_set, capacities_num, \
                "Activation probability in layer 1", "Activation probabiltiy in layer 2", "Numerical capacity", "The numerical capacity of the two-layer network", ax=axes[1,1])

    # plot the scatter of the lower branch:
    midpoint = int(len(perc_act_L2_set)/2)
    # scatter_plot_color(capacities_num[0:midpoint,:].flatten(), robustness_score[0:midpoint,:].flatten(), hipp_index_score[0:midpoint,:].flatten(), \
    #                    "Memory capacity", "Robustness score", "Hippo index score", "The relationship between the robustness score and the hippo index score", ax=axes[2,0], square=False)

    # scatter_plot_color(information_capacity_num[0:midpoint,:].flatten(), robustness_score[0:midpoint,:].flatten(), hipp_index_score[0:midpoint,:].flatten(), \
    #                    "Information capacity", "Robustness score", "Hippo index score", "The relationship between the robustness score and the hippo index score", ax=axes[2,1], square=False)

    utils_num.scatter_plot_color(np.concat((capacities_theoretical[0:midpoint,:].flatten(), capacities_ideal.flatten())), np.concat((robustness_score[0:midpoint,:].flatten(), robustness_score_ideal.flatten())), np.concat((hipp_index_score[0:midpoint,:].flatten(), hipp_index_score_ideal.flatten())), \
                        "Memory capacity", "Robustness score", "Hippo index score", "The relationship between the robustness score and the hippo index score with ideal hippocampal index network", ax=axes[2,0], square=False)

    utils_num.scatter_plot_color(np.concat((information_capacity_theoretical[0:midpoint,:].flatten(), information_capacities_ideal.flatten())), np.concat((robustness_score[0:midpoint,:].flatten(), robustness_score_ideal.flatten())), np.concat((hipp_index_score[0:midpoint,:].flatten(), hipp_index_score_ideal.flatten())), \
                        "Information capacity", "Robustness score", "Hippo index score", "The relationship between the robustness score and the hippo index score with ideal hippocampal index network", ax=axes[2,1], square=False)
    Info_times_memory_theoretical = information_capacity_theoretical*capacities_theoretical
    normalization_factor = np.max(Info_times_memory_theoretical)

    Info_times_memory_theoretical = Info_times_memory_theoretical/normalization_factor
    Info_times_memory_ideal = information_capacities_ideal*capacities_ideal/normalization_factor

    # scatter_plot_color(np.concat((Info_times_memory_theoretical[0:midpoint,:].flatten(), Info_times_memory_ideal.flatten())), np.concat((robustness_score[0:midpoint,:].flatten(),\
    #                      robustness_score_ideal.flatten())), np.concat((hipp_index_score[0:midpoint,:].flatten(), hipp_index_score_ideal.flatten())), \
    #                      "Information times memory", "Robustness score", "Hippo index score", "The relationship between the robustness score and the hippo index score with ideal hippocampal index network", ax=axes[2,2], square=False)

    utils_num.scatter_plot_color(Info_times_memory_theoretical[0:midpoint,:].flatten(), robustness_score[0:midpoint,:].flatten(), hipp_index_score[0:midpoint,:].flatten(), \
                        "Information times memory", "Robustness score", "Hippo index score", "Info times memory theoretical", ax=axes[2,2], square=False)
    # set the color bar ticks:
    cbar = plt.colorbar(axes[2,2].collections[0], ax=axes[2,2])
    cbar.set_ticks(np.arange(0.5, 0.61, 0.02))
    # scatter_plot_color(Info_times_memory_ideal, robustness_score_ideal, hipp_index_score_ideal, \
    #                         "Information times memory", "Robustness score", "Hippo index score", "", ax=axes[2,2], square=False)
    plt.sca(axes[2,2])
    plt.scatter(Info_times_memory_ideal, robustness_score_ideal, c=hipp_index_score_ideal, cmap='cool', vmin=0, vmax=1, s=12)

    axes[2,2].set_aspect(2.5)

    utils.print_parameters_on_plot(axes[0,2], pars)

    # import matplotlib
    # matplotlib.use('TkAgg')

    plt.show(block=False)
    plt.savefig('results/robustness_score_two_layer_V7_ablation=001 k = 0.0, n_layer1 = 150, n_layer_2 = 450'+utils_num.make_name_with_time()+'.png')
    plt.savefig('results/robustness_score_two_layer_V7_ablation=001 k = 0.0, n_layer1 = 150, n_layer_2 = 450'+utils_num.make_name_with_time()+'.pdf')

    print("end")


