import numpy as np
from numpy import random
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import scipy
from random_connection_network import random_connect_network
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




# make example from concepts:
def make_example(concepts, n_example_per_concept, coding_level, reinit_prop=0.9):
    # correlation = 1- reinit_prop +(1-2p)**2
    n_concept = len(concepts)
    n_neuron = len(concepts[0])
    examples = []
    for i in range(n_concept):
        for j in range(n_example_per_concept):
            example = copy.deepcopy(concepts[i])
            reinit = torch.bernoulli(torch.ones(n_neuron)*reinit_prop)
            new_values = torch.bernoulli(coding_level)*2-1
            example = example * (1-reinit) + reinit * new_values
            examples.append(example)
    examples = torch.stack(examples, dim=0).float()
    return examples

#make a parallel version of the code to speed up the calculation
def calculate_errors(n_concept, n_example_per_concept, mask, neuron_act_prop, pars):
    weight_std_init = pars["weight_std_init"]
    neuron_base_state = pars["neuron_base_state"]
    W_symmetric = pars["W_symmetric"]
    reinit_prop = pars["reinit_prop"]
    use_reg = pars["use_reg"]
    kappa = pars["kappa"]
    n_neuron = pars["n_neuron"]


    # make the concepts and examples
    concepts_all = utils.make_pattern(n_concept, n_neuron, perc_active=neuron_act_prop)
    examples_all = make_example(concepts_all, n_example_per_concept, coding_level = neuron_act_prop, reinit_prop=reinit_prop)
    # examples_all = torch.from_numpy(examples_all).float()
    if type(concepts_all) == np.ndarray:
        concepts_all = torch.from_numpy(concepts_all).float()

    # init the network
    network = random_connect_network(n_neuron, W_symmetric=W_symmetric,connection_prop=None, weight_std_init=weight_std_init, mask=mask, neuron_base_state=neuron_base_state)
    success, stored_patterns_all = network.train_svm(examples_all, use_reg = use_reg)

    # calculate the error
    error_example = 1-utils.network_score(network.weight, network.b, examples_all, kappa = kappa)
    error_concept = 1-utils.network_score(network.weight, network.b, concepts_all, kappa = kappa)
    return error_concept.item(), error_example.item()

if __name__ == '__main__':

    sys.argv += "--n_neuron 100 --n_repeat 10 --reinit_prop 0.3 --pattern_act_mean 0.25  --ratio_conn_mean 0.5 --change_in_degree\
        --heter_type both_positive_corr --n_process 8".split()

    parser = argparse.ArgumentParser(description="Memorize correlated concepts with one layer network")
    parser.add_argument('--n_neuron', type=int, default=50, help='number of neurons')
    parser.add_argument('--n_repeat', type=int, default=10, help='number of repeats')
    parser.add_argument('--kappa', type=float, default=0, help='kappa for the SVM')
    parser.add_argument('--use_reg', action='store_true', help='whether to use kappa')
    parser.add_argument('--weight_std_init', type=float, default=0.1, help='initial weight std')
    parser.add_argument('--neuron_base_state', type=int, default=-1, help='neuron base state')
    parser.add_argument('--W_symmetric', action='store_true', help='whether the weights are symmetric')
    parser.add_argument('--reinit_prop', type=float, default=1.0, help='reinit prop: 1: total random, 0: no random')
    parser.add_argument('--pattern_act_mean', type=float, default=0.25, help='mean of the neuron activation')
    parser.add_argument('--pattern_act_std', type=float, default=0.0, help='std of the neuron activation')
    parser.add_argument('--ratio_conn_mean', type=float, default=0.5, help='mean of the connection ratio')
    parser.add_argument('--ratio_conn_std', type=float, default=0.0, help='std of the connection ratio')
    parser.add_argument('--change_in_degree', action='store_true', help='change the in degree of the neurons')
    parser.add_argument('--change_out_degree', action='store_true', help='change the out degree of the neurons')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # parser.add_argument('--use_coding_level_heter', action='store_true', help='whether to use heterogeneous coding level, default is heterogeneous network topology')
    parser.add_argument('--heter_type', type=str, default='ratio_conn_std', choices=['coding_level', 'network_connection', 'both_positive_corr', 'both_negative_corr', 'both_uncorr'], help='heterogeneity type')
    parser.add_argument('--n_process', type=int, default=16, help='number of processes for parallelization')
    args = parser.parse_args()

    pars = vars(args)




    n_neuron = pars['n_neuron']
    n_repeat = pars['n_repeat']
    kappa = pars['kappa']
    use_reg = pars['use_reg']
    weight_std_init = pars['weight_std_init']
    neuron_base_state = pars['neuron_base_state']
    W_symmetric = pars['W_symmetric']
    reinit_prop = pars['reinit_prop']

    pattern_act_mean = pars['pattern_act_mean']
    pattern_act_std = pars['pattern_act_std']
    ratio_conn_mean = pars['ratio_conn_mean']
    ratio_conn_std = pars['ratio_conn_std']
    heter_type = pars['heter_type']


    seed = pars['seed']
    n_process = np.min([mp.cpu_count(), pars["n_process"]])

    # set the seed
    random.seed(seed)


    # use log scale for the number of concepts and examples
    n_concept_set = np.geomspace(1, n_neuron*2, 20).astype(int)
    n_example_per_concept_set = np.geomspace(1, n_neuron*2, 21).astype(int)
    n_heter_level = 3
    network_heter_level_set = np.linspace(0.0, 0.18, n_heter_level)
    coding_level_heter_set = np.linspace(0.0, 0.14, n_heter_level)

    # if use_coding_level_heter:
    #     heter_level_set = coding_level_heter_set
    # else:
    #     heter_level_set = network_heter_level_set

    # step =5
    # n_concept_set = np.arange(1, n_neuron*2, step)
    # n_example_per_concept_set = np.arange(1, n_neuron, step)
    

    concept_errors = np.zeros((len(n_concept_set), len(n_example_per_concept_set), n_heter_level, n_repeat))
    example_errors = np.zeros((len(n_concept_set), len(n_example_per_concept_set), n_heter_level, n_repeat))

    # parallelize the code
    # mp.set_start_method("spawn", force=True)
    print(f"Start parallel computation, with {n_process} processes")
    t_start = time.time()

    with mp.Pool(processes=n_process) as pool:
        # pool_result_object is a 4D list
        pool_results_object = [[[[[] for m in range(n_repeat)] for k in range(n_heter_level)] for j in range(len(n_example_per_concept_set))] for i in range(len(n_concept_set))]
        for m in range(n_heter_level):
            # make the mask for the network
            # uniform distribution
            if heter_type == 'network_connection':
                ratio_conn_std = network_heter_level_set[m]
                pattern_act_std = pars['pattern_act_std']
            elif heter_type == 'coding_level':
                pattern_act_std = coding_level_heter_set[m]
                ratio_conn_std = pars['ratio_conn_std']
            elif heter_type == 'both_positive_corr' or heter_type == 'both_negative_corr' or heter_type == 'both_uncorr':
                pattern_act_std = coding_level_heter_set[m]
                ratio_conn_std = network_heter_level_set[m]

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


            for i, n_concept in enumerate(n_concept_set):
                for j, n_pattern_per_concept in enumerate(n_example_per_concept_set):
                    concept_error_set = np.zeros(n_repeat)*np.nan
                    example_error_set = np.zeros(n_repeat)*np.nan
                    for k in range(n_repeat):
                        pool_results_object[i][j][m][k] = pool.apply_async(calculate_errors, args=(n_concept, n_pattern_per_concept, mask, neuron_act_prop, pars)) 
                        print("n_concept: ", n_concept, "n_example_per_concept: ", n_pattern_per_concept, "network heterogeneity: ", ratio_conn_std, "repeat: ", k)
                        # for debugging: 
                        # error_concept, error_example = calculate_errors(n_concept, n_pattern_per_concept, mask, neuron_act_prop, pars)
                        # concept_errors[i,j,m,k] = error_concept
                        # concept_errors[i,j,m,k] = error_concept
                        # print("n_concept: ", n_concept, "n_example_per_concept: ", n_pattern_per_concept, "network heterogeneity: ", ratio_conn_std, "repeat: ", k, "concept error: ", error_concept, "example error: ", error_example)
                        
        # fetch the parallel results
        for m in range(n_heter_level):
            ratio_conn_std = network_heter_level_set[m]
            for i, n_concept in enumerate(n_concept_set):
                for j, n_pattern_per_concept in enumerate(n_example_per_concept_set):
                    for k in range(n_repeat):
                        try:
                            error_concept, error_example = pool_results_object[i][j][m][k].get()
                        except:
                            raise ValueError("Error in getting the results of pool_results_object[%d][%d][%d][%d]" % (i,j,m,k))
                        
                        concept_errors[i,j,m,k] = error_concept
                        example_errors[i,j,m,k] = error_example
                        
                    print("n_concept: ", n_concept, "n_example_per_concept: ", n_pattern_per_concept, "concept error: ", "network heterogeneity: ", ratio_conn_std, \
                        "example error: ", np.mean(example_errors[i,j,m,:]), "concept error: ", np.mean(concept_errors[i,j,m,:]))
    
    pool.close()
    pool.join()

    t_end = time.time()
    print(f"Parallel computation finished, time used: {t_end-t_start} seconds")

    example_errors = np.nanmean(example_errors, axis=3)
    concept_errors = np.nanmean(concept_errors, axis=3)
    # save the results for later use
    results = {}
    results['concept_errors'] = concept_errors
    results['example_errors'] = example_errors
    results['n_concept_set'] = n_concept_set
    results['n_example_per_concept_set'] = n_example_per_concept_set
    results['network_heter_level_set'] = network_heter_level_set
    results['coding_level_heter_set'] = coding_level_heter_set
    results['pars'] = pars

    with open('results/training_error_concept_and_example_3D_matrix'+utils.make_name_with_time()+'.pkl', 'wb') as f:
        pickle.dump(results, f)







