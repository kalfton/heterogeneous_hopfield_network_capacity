import numpy as np
from numpy import random
import torch
from random_connection_network import random_connect_network
import utils_basic as utils
import utils_numerical as utils_num
import copy
import pickle
import warnings
import argparse
import multiprocessing as mp
from theoretical_two_layer_network import theoretical_two_layer_network_capacity
from plot_robustness_analysis import plot_robustness_analysis


def calc_robustness_score(network: random_connect_network, stored_patterns, ablation_unit_matrix, N_feature_layer):
    """
    Compute the robustness with respect to the removal of specific units in the memory layer.
    """
    retrieval_pattern = network.forward(stored_patterns*ablation_unit_matrix,  n_step = 1, dt_train = 1)
    retrieval_pattern_feature = (retrieval_pattern[:,:N_feature_layer]).numpy()
    stored_patterns_feature = (stored_patterns[:,:N_feature_layer]).numpy()

    robustness_by_pattern = np.zeros(stored_patterns_feature.shape[0])
    for mu in range(stored_patterns_feature.shape[0]): # for each pattern, we can calculate the similarity score, then average it over all patterns.
        robustness_by_pattern[mu] = utils.calc_similarity_score(stored_patterns_feature[mu], retrieval_pattern_feature[mu])
    robustness = np.mean(robustness_by_pattern) # average over all patterns, this is the final robustness score.
    
    return robustness.item()

def network_robustness(network: random_connect_network, stored_patterns, N_feature_layer, N_neuron, N_removal=1):
    """
    Compute the robustness of the two-layer network with respect to the removal of active unit in the memory layer.
    """ 
    network= copy.deepcopy(network)
    N_neuron = network.n_neuron
    N_memory_layer = N_neuron - N_feature_layer
    repeatition = N_memory_layer

    # Compute the number of nodes to remove
    num_nodes_to_remove = int(N_removal)

    # Calculate the stability condition after removing one node
    repeatition = N_memory_layer

    # Remove nodes
    robustness_all = []
    for i in range(repeatition):
        ablation_units_matrix = torch.ones_like(stored_patterns) # the ablation matrix represents the units that should be removed for each pattern, 0 means the unit is removed, 1 means the unit is kept. 
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

    return robustness.item(), torch.nan

def calculate_robustness_and_index_score(capacity, perc_act_L1, perc_act_L2, pars):
    # parameters:
    n_neuron = pars["n_neuron"]
    n_neuron_L2 = pars["n_neuron_L2"]
    neuron_base_state = pars["neuron_base_state"]
    network_type = pars["network_type"]
    W_symmetric = pars["W_symmetric"]


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
        if pars["use_reg"]:
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

    parser = argparse.ArgumentParser(description='Two layer network finite size analysis')
    parser.add_argument('--method', type=str, default='svm', help='training method',
    choices=['PLA', 'svm'])
    parser.add_argument('--n_neuron', type=int, default=150, help='number of neurons in layer 1')
    parser.add_argument('--n_neuron_L2', type=int, default=450, help='number of neurons in layer 2')
    parser.add_argument('--W_symmetric', action='store_true', help='whether the W matrix is symmetric')
    parser.add_argument('--training_max_iter', type=int, default=10000, help='maximum number of iterations for training')
    parser.add_argument('--weight_std_init', type=float, default=0.1, help='standard deviation of the initial weights')
    parser.add_argument('--lr_PLA', type=float, default=0.02, help='learning rate for PLA')
    parser.add_argument('--network_type', type=str, default='RBM', help='type of network')
    parser.add_argument('--neuron_base_state', type=float, default=-1, help='the base state of the neurons')
    parser.add_argument('--epsilon', type=float, default=0.01, help='the error threshold for the capacity')
    parser.add_argument('--use_reg', action='store_true', help='whether to use kappa')
    parser.add_argument('--kappa', type=float, default=0.0, help='the kappa for the training')
    parser.add_argument('--n_process', type=int, default=10, help='number of processes to run in parallel')
    parser.add_argument('--N_removal', type=float, default=1, help='the number of neurons to remove')

    args = parser.parse_args()
    pars = vars(args)

    # parameters:
    n_neuron = pars["n_neuron"]
    n_neuron_L2 = pars["n_neuron_L2"]
    neuron_base_state = pars["neuron_base_state"]
    network_type = pars["network_type"]
    W_symmetric = pars["W_symmetric"]
    use_reg = pars["use_reg"]
    kappa = pars["kappa"]
    N_removal = pars["N_removal"]


    weight_std_init = pars["weight_std_init"]
    method = pars["method"]
    lr_PLA = pars["lr_PLA"]
    training_max_iter = pars["training_max_iter"]
    n_process = min(pars["n_process"], mp.cpu_count())

    perc_act_L1_set = np.concatenate((np.arange(0.01, 0.0999, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.999, 0.03)))
    perc_act_L2_set = np.concatenate((np.arange(0.01, 0.0999, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.999, 0.03)))

    capacities_theoretical, information_capacity_theoretical, _, _ = theoretical_two_layer_network_capacity(perc_act_L1_set, perc_act_L2_set, n_neuron, n_neuron_L2, use_reg, kappa, make_plot=False)

    capacities = (np.floor(capacities_theoretical*(n_neuron+n_neuron_L2))).astype(int)
    information_capacity = information_capacity_theoretical*(n_neuron+n_neuron_L2)**2


    robustness_score = np.zeros((len(perc_act_L2_set), len(perc_act_L1_set)))
    hipp_index_score = np.zeros((len(perc_act_L2_set), len(perc_act_L1_set)))


    with mp.Pool(n_process) as pool:
        pool_results_object = [[[] for i in range(len(perc_act_L2_set))] for j in range(len(perc_act_L1_set))]
        for i in range(len(perc_act_L2_set)):
            for j in range(len(perc_act_L1_set)):
                print("perc_act_L1 = %f, perc_act_L2 = %f, capacity = %f" % (perc_act_L1_set[j], perc_act_L2_set[i], capacities[i][j]))
                pool_results_object[i][j] = pool.apply_async(calculate_robustness_and_index_score, args=(capacities[i][j], perc_act_L1_set[j], perc_act_L2_set[i], pars))

        for i in range(len(perc_act_L2_set)):
            for j in range(len(perc_act_L1_set)):
                try:
                    robustness_score[i,j], hipp_index_score[i,j] = pool_results_object[i][j].get()
                except:
                    raise ValueError("Error in getting the results of pool_results_object[%d][%d]" % (i,j))
            
                print("perc_act_L1 = %f, perc_act_L2 = %f, capacity = %f, robustness_score = %f, hipp_index_score = %f" % (perc_act_L1_set[j], perc_act_L2_set[i], capacities[i][j], robustness_score[i,j], hipp_index_score[i,j]))


    # make a network that use ideal hippo index theory to store memory and its robustness score:
    mask = torch.ones((n_neuron+n_neuron_L2, n_neuron+n_neuron_L2))
    mask[:n_neuron, :n_neuron] = 0
    mask[n_neuron:, n_neuron:] = 0
    mask.fill_diagonal_(0)

    robustness_score_ideal = np.zeros(len(perc_act_L1_set))
    hipp_index_score_ideal = np.zeros(len(perc_act_L1_set))
    capacities_ideal = np.zeros(len(perc_act_L1_set))
    for j in range(len(perc_act_L1_set)):
        n_pattern = n_neuron_L2
        patterns_L1 = utils.make_pattern(n_pattern, n_neuron, perc_active=perc_act_L1_set[j], neuron_base_state=neuron_base_state)
        # remove replicate patterns:
        patterns_L1 = torch.unique(patterns_L1, dim=0)
        if patterns_L1.shape[0] < n_pattern:
            print(f"n_pattern = {n_pattern}, but the number of unique patterns is {patterns_L1.shape[0]}, skip this case.")
            robustness_score_ideal[j] = np.nan
            hipp_index_score_ideal[j] = np.nan
            capacities_ideal[j] = np.nan
            continue
        patterns_L2 = torch.diag(torch.ones(n_neuron_L2))*2-1
        patterns_L2 = patterns_L2[:n_pattern,:]
        patterns_all = torch.cat((patterns_L1, patterns_L2), dim=1)
        

        network = random_connect_network(n_neuron+n_neuron_L2, W_symmetric=W_symmetric,connection_prop=None, weight_std_init=weight_std_init, mask=mask, neuron_base_state=neuron_base_state)
        cross_weight = torch.zeros((n_neuron, n_neuron_L2))
        for i in range(n_neuron_L2):
            cross_weight[:, i] = patterns_L1[i]
        network.weight = torch.zeros((n_neuron+n_neuron_L2, n_neuron+n_neuron_L2))
        network.weight[:n_neuron, n_neuron:] = cross_weight
        network.weight[n_neuron:, :n_neuron] = cross_weight.T
        
        bias_L1 = torch.zeros(n_neuron)
        for i in range(n_neuron_L2):
            bias_L1 = -(torch.max(torch.matmul(cross_weight, patterns_L2.T), dim=1)[0]-0.5)
        bias_L2 = -(torch.max(torch.matmul(cross_weight.T, patterns_L1.T), dim=1)[0]-0.5)
        network.b = torch.cat((bias_L1, bias_L2), dim=0)
        # check that the patterns are stored correctly:
        stored_patterns_all = network.forward(patterns_all, n_step = 1, dt_train = 1)
        if torch.sum(stored_patterns_all*patterns_all <0 ) > 1:
            print(f"The stored patterns are not correct! perc_act_L1 = {perc_act_L1_set[j]}")
        (robustness_score_ideal[j],_)= network_robustness(network, patterns_all, n_neuron, n_neuron_L2, N_removal=N_removal)

        W_12_trained = network.weight.clone().numpy()[:n_neuron, n_neuron:]
        W_21_trained = network.weight.clone().numpy()[n_neuron:, :n_neuron]
        hipp_index_score_ideal[j] = utils_num.index_analysis_v4(W_12_trained, W_21_trained.T, patterns_L1=patterns_L1,patterns_L2=patterns_L2, perc_active_L1=perc_act_L1_set[j], neuron_base_state = neuron_base_state, makeplot=False)

        capacities_ideal[j] = n_neuron_L2

    capacities_ideal = capacities_ideal/(pars["n_neuron"]+pars["n_neuron_L2"])
    information_capacity_ideal = np.zeros_like(capacities_ideal)
    for j in range(len(perc_act_L1_set)):
        information_capacity_ideal[j] = capacities_ideal[j]*utils.information_entropy(perc_act_L1_set[j])*pars["n_neuron"]/(pars["n_neuron"]+pars["n_neuron_L2"])
    
    # save the results:
    results = {}
    results["robustness_score"] = robustness_score
    results["hipp_index_score"] = hipp_index_score
    results["capacities_theoretical"] = capacities_theoretical
    results["information_capacity_theoretical"] = information_capacity_theoretical

    results["robustness_score_ideal"] = robustness_score_ideal
    results["hipp_index_score_ideal"] = hipp_index_score_ideal
    results["capacities_ideal"] = capacities_ideal
    results["information_capacity_ideal"] = information_capacity_ideal

    results["perc_act_L1_set"] = perc_act_L1_set
    results["perc_act_L2_set"] = perc_act_L2_set
    results["pars"] = pars
    
    with open(f'results/robustness_score_two_layer_ablation=1, n_layer1 = {n_neuron}, n_layer_2 = {n_neuron_L2}.pkl', 'wb') as f:
        pickle.dump(results, f)

    # # plot the results:
    plot_robustness_analysis(robustness_score, hipp_index_score, information_capacity_theoretical, capacities_theoretical,\
                             robustness_score_ideal, hipp_index_score_ideal, capacities_ideal, information_capacity_ideal, perc_act_L1_set, perc_act_L2_set, pars)


