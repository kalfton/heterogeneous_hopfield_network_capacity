import numpy as np
from numpy import random
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import scipy
from random_connection_network import random_connect_network
from committee_machine_network_V4 import random_connect_network as random_connect_network_committee
from utils_theoretical import theoretical_network_capacity
from utils_theoretical import theoretical_network_capacity_kappa
import utils_basic as utils
import pickle
import warnings
import time
import argparse
import sys

import multiprocessing as mp

def calculate_numerical_cacpacity(pars, max_n_pattern:int, step:int, epsilon, seed1=None, seed2=None, network=None, neuron_act_prop=None, connection_prop=None):

    # parameters:
    n_neuron = pars["n_neuron"]
    training_max_iter = pars["training_max_iter"]
    method = pars["method"]
    lr_PLA = pars["lr_PLA"]
    W_symmetric = not pars["W_notsymmetric"]
    weight_std_init = pars["weight_std_init"]
    use_reg = pars["use_reg"]
    kappa = pars["kappa"]

    use_committee_machine = pars["use_committee_machine"]
    n_dendrites = pars["n_dentrates"]
    lr_committee = pars["lr_committee"]
    add_bias = pars["add_bias"]

    if seed1 is not None and seed2 is not None:
        random.seed(seed1)
        torch.manual_seed(seed2)
    # Binary search:
    # there is not enough patterns to be stored: lower the max_n_pattern
    if neuron_act_prop is None:
        raise ValueError("neuron_act_prop is None, please specify the neuron_act_prop")

    patterns = utils.make_pattern(max_n_pattern, n_neuron, perc_active=neuron_act_prop)
    # eliminate the repeating patterns:
    patterns = torch.unique(patterns, dim=0)


    n_pattern_unique = patterns.shape[0]
    left, right = 1, n_pattern_unique

    if network is None:
        if connection_prop is None:
            raise ValueError("connection_prop is None, please specify the connection_prop")
        if use_committee_machine:
            network = random_connect_network_committee(n_neuron, connection_prop=connection_prop, weight_std_init=weight_std_init, hidden_size=n_dendrites, add_bias=add_bias)
        else:
            network = random_connect_network(n_neuron, W_symmetric=W_symmetric,connection_prop=connection_prop, weight_std_init=weight_std_init)
    
    while left <= right:
        # make the binary search uneven, because when it is under capacity, the calculation is faster:
        mid = int(0.75*left+0.25*right)  #(left + right) // 2
        n_pattern = mid
        patterns = utils.make_pattern(n_pattern, n_neuron, perc_active=neuron_act_prop)

        # # digitizing the original patterns:
        # patterns_all_d=torch.zeros_like(patterns)+0.5
        # patterns_all_d[patterns>0.7] = 1
        # patterns_all_d[patterns<0.3] = 0

        network.reinitiate()
        if use_committee_machine:
            success, stored_patterns = network.train(patterns, training_max_iter=training_max_iter, lr=lr_committee, method=method)
        else:
            if method.lower() == 'PLA'.lower():
                success, stored_patterns = network.train_PLA(patterns, training_max_iter=training_max_iter, lr=lr_PLA)
            elif method.lower() == 'svm'.lower():
                if use_reg:
                    success, stored_patterns = network.train_svm(patterns, kappa=kappa)
                else:
                    success, stored_patterns = network.train_svm(patterns, use_reg=False)

        # # calculate the mean error of retrieval:
        # # digitizing the stored patterns:
        # stored_patterns_all_d=torch.zeros_like(stored_patterns)+0.5
        # stored_patterns_all_d[stored_patterns>0.7] = 1
        # stored_patterns_all_d[stored_patterns<0.3] = 0

        # # digitizing the original patterns:
        # patterns_all_d=torch.zeros_like(patterns)+0.5
        # patterns_all_d[patterns>0.7] = 1
        # patterns_all_d[patterns<0.3] = 0

        # # calculate the mean error:
        # mean_error = torch.mean(torch.abs(stored_patterns_all_d-patterns_all_d))
        if use_committee_machine:
            mean_error = ((stored_patterns*patterns)<=0).to(torch.float).mean()
        else:
            mean_error = 1-utils.network_score(network.weight, network.b, patterns, kappa = kappa)
        
        if mean_error>epsilon: # the capacity is smaller than n_pattern
            right = mid - step
        else: # the capacity is larger than n_pattern
            left = mid + step

    return left

def cacpacity_for_parallel(pars_copy, seed1, seed2):
    # parameters:
    n_neuron = pars_copy["n_neuron"]
    epsilon = pars_copy["epsilon"]
    use_committee_machine = pars_copy["use_committee_machine"]
    n_dendrites = pars_copy["n_dentrates"]
    add_bias = pars_copy["add_bias"]

    if pars_copy["distribution"] == "uniform":
        neuron_act_prop = (torch.rand((n_neuron,))-0.5)*(pars_copy["pattern_act_std"]*np.sqrt(12))+pars_copy["pattern_act_mean"]
        if pars_copy["change_in_degree"]:
            connection_prop_in = (torch.rand((n_neuron,))-0.5)*(pars_copy["ratio_conn_std"]*np.sqrt(12))+pars_copy["ratio_conn_mean"]
        else:
            connection_prop_in = torch.ones(n_neuron)*pars_copy["ratio_conn_mean"]
        if pars_copy["change_out_degree"]:
            connection_prop_out = (torch.rand((n_neuron,))-0.5)*(pars_copy["ratio_conn_std"]*np.sqrt(12))+pars_copy["ratio_conn_mean"]
        else:
            connection_prop_out = torch.ones(n_neuron)*pars_copy["ratio_conn_mean"]
        if torch.any(neuron_act_prop>pars_copy["act_prob_range"][1]) or torch.any(neuron_act_prop<pars_copy["act_prob_range"][0]):
            warnings.warn("The neuron activation probability is out of the range ")
        if torch.any(torch.cat((connection_prop_in, connection_prop_out))>1-0.01) or torch.any(torch.cat((connection_prop_in, connection_prop_out))<0.01):
            warnings.warn("The connection probability is out of the range ")
    elif pars_copy["distribution"] == "normal":
        neuron_act_prop = torch.normal(pars_copy["pattern_act_mean"], pars_copy["pattern_act_std"], (n_neuron,))
        neuron_act_prop = torch.clip(neuron_act_prop, pars_copy["act_prob_range"][0], pars_copy["act_prob_range"][1]) # (0.01, 0.4-0.01)
        if pars_copy["change_in_degree"]:
            connection_prop_in = torch.normal(pars_copy["ratio_conn_mean"], pars_copy["ratio_conn_std"], (n_neuron,)) 
            connection_prop_in = torch.clip(connection_prop_in, pars_copy["conn_prob_range"][0], pars_copy["conn_prob_range"][1]) # (0.1, 1-0.1)
        else:
            connection_prop_in = torch.ones(n_neuron)*pars_copy["ratio_conn_mean"]
        if pars_copy["change_out_degree"]:
            connection_prop_out = torch.normal(pars_copy["ratio_conn_mean"], pars_copy["ratio_conn_std"], (n_neuron,))
            connection_prop_out = torch.clip(connection_prop_out, pars_copy["conn_prob_range"][0], pars_copy["conn_prob_range"][1]) # (0.1, 1-0.1)
        else:
            connection_prop_out = torch.ones(n_neuron)*pars_copy["ratio_conn_mean"]

    # control the mean of connetion prop if needed:
    # connection_prop = connection_prop - torch.mean(connection_prop) + pars_copy["ratio_conn_mean"]

    # The capacity when the heterogeneity of activation prob and connection prob are independent:
    # generate the connection mask:
    mask_prob = (connection_prop_in[:, None] * connection_prop_out[None,:])/pars_copy["ratio_conn_mean"]
    mask_prob = torch.clip(mask_prob, 0, 1)
    mask = torch.bernoulli(mask_prob)
    mask.fill_diagonal_(0) # important: no self-connection
    
    if use_committee_machine:
        network = random_connect_network_committee(n_neuron, mask = mask, weight_std_init=pars_copy["weight_std_init"], hidden_size=n_dendrites, add_bias=add_bias)
    else:
        network = random_connect_network(n_neuron, W_symmetric=not pars_copy["W_notsymmetric"], mask = mask, weight_std_init=pars_copy["weight_std_init"])

    cap = calculate_numerical_cacpacity(pars_copy, n_neuron*7, 2, epsilon=epsilon, seed1=seed1, seed2=seed2, neuron_act_prop=neuron_act_prop, network=network)
    capacity = cap/n_neuron
    print("Ratio_coon_std, %f, pattern_act_std, %f, The numerical capacity is: %f" % (pars_copy["ratio_conn_std"], pars_copy["pattern_act_std"], capacity))

    # theoretical capacity
    if pars_copy["use_reg"] and not use_committee_machine:
        capacity_theoretical = theoretical_network_capacity_kappa(network.weight.detach().numpy(), neuron_act_prop.detach().numpy(), pars_copy["kappa"])
    elif not use_committee_machine:
        capacity_theoretical = theoretical_network_capacity(network.weight.detach().numpy(), neuron_act_prop.detach().numpy())
    else:
        capacity_theoretical = np.nan
    print("Ratio_coon_std, %f, pattern_act_std, %f, The theoretical capacity is: %f" % (pars_copy["ratio_conn_std"], pars_copy["pattern_act_std"], capacity_theoretical))

    # The capcity when the heterogeneity of activation prob and connection prob are positively correlated:
    # sort the neuron_act_prop according to the connection_prop:
    neuron_act_prop_in_order = neuron_act_prop.sort()[0]

    if pars_copy["change_in_degree"]:
        connection_prop_in_in_order, sorted_ind = connection_prop_in.sort()
    elif pars_copy["change_out_degree"]:
        connection_prop_out_in_order, sorted_ind = connection_prop_out.sort()
    else:
        raise ValueError("The connection prop is not changed, please specify the connection prop to be changed")
    
    # sort the neuron_act_prop to match the connection_prop:
    recover_ind = torch.argsort(sorted_ind)
    neuron_act_prop_matched = neuron_act_prop_in_order[recover_ind]
    # connection_prop_matched = connection_prop_in_order[recover_ind]

    if use_committee_machine:
        network = random_connect_network_committee(n_neuron, mask = mask, weight_std_init=pars_copy["weight_std_init"], hidden_size=n_dendrites, add_bias=add_bias)
    else:
        network = random_connect_network(n_neuron, W_symmetric=not pars_copy["W_notsymmetric"], mask = mask, weight_std_init=pars_copy["weight_std_init"])
    cap = calculate_numerical_cacpacity(pars_copy, n_neuron*7, 5, epsilon=epsilon, seed1=seed1, seed2=seed2, neuron_act_prop=neuron_act_prop_matched, network=network)
    capacity_matched = cap/n_neuron
    # theoretical capacity
    if pars_copy["use_reg"] and not use_committee_machine:
        capacity_theoretical_matched = theoretical_network_capacity_kappa(network.weight.detach().numpy(), neuron_act_prop.detach().numpy(), pars_copy["kappa"])
    elif not use_committee_machine:
        capacity_theoretical_matched = theoretical_network_capacity(network.weight.detach().numpy(), neuron_act_prop.detach().numpy())
    else:
        capacity_theoretical_matched = np.nan

    
    # The capcity when the heterogeneity of activation prob and connection prob are negatively correlated:
    if use_committee_machine:
        network = random_connect_network_committee(n_neuron, mask = mask, weight_std_init=pars_copy["weight_std_init"], hidden_size=n_dendrites, add_bias=add_bias)
    else:
        network = random_connect_network(n_neuron, W_symmetric=not pars_copy["W_notsymmetric"], mask = mask, weight_std_init=pars_copy["weight_std_init"])
    neuron_act_prop_anti_matched = torch.flip(neuron_act_prop_in_order, dims=[0])[recover_ind]
    cap = calculate_numerical_cacpacity(pars_copy, n_neuron*7, 5, epsilon=epsilon, seed1=seed1, seed2=seed2, neuron_act_prop=neuron_act_prop_anti_matched, network=network)
    capacity_anti_matched = cap/n_neuron
    # theoretical capacity
    # theoretical capacity
    if pars_copy["use_reg"] and not use_committee_machine:
        capacity_theoretical_anti_matched = theoretical_network_capacity_kappa(network.weight.detach().numpy(), neuron_act_prop.detach().numpy(), pars_copy["kappa"])
    elif not use_committee_machine:
        capacity_theoretical_anti_matched = theoretical_network_capacity(network.weight.detach().numpy(), neuron_act_prop.detach().numpy())
    else:
        capacity_theoretical_anti_matched = np.nan

    return capacity, capacity_theoretical, capacity_matched, capacity_theoretical_matched, capacity_anti_matched, capacity_theoretical_anti_matched

if __name__ == '__main__':

    # sys.argv += "--W_notsymmetric --n_neuron 100 --method backprop --lr_committee 0.05 --change_in_degree --n_repeat 10 --n_process 15  --pattern_act_mean 0.25 --use_committee_machine\
    #     --training_max_iter 1000 --n_dentrates 3 --add_bias".split()
    # # --lr_committee 0.05, training_max_iter 1000 for backprop  and --lr_committee 0.2 for ALA

    parser = argparse.ArgumentParser(description='Train neural dependency parser in pytorch')
    parser.add_argument('--method', type=str, default='svm', help='training method', choices=['PLA', 'svm', 'ALA', 'backprop'])
    parser.add_argument('--n_neuron', type=int, default=500, help='number of neurons in the first layer')
    parser.add_argument('--pattern_act_mean', type=float, default=0.25, help='mean of the pattern activation probability')
    parser.add_argument('--pattern_act_std', type=float, default=0.1, help='std of the pattern activation probability')
    parser.add_argument('--ratio_conn_mean', type=float, default=0.5, help='mean of the connection probability')
    parser.add_argument('--ratio_conn_std', type=float, default=0.2, help='std of the connection probability')
    parser.add_argument('--weight_std_init', type=float, default=0.1, help='std of the initial weight')
    parser.add_argument('--W_notsymmetric', action='store_true', help='whether the weight matrix is not symmetric')
    parser.add_argument('--training_max_iter', type=int, default=10000, help='max number of iterations for training')
    parser.add_argument('--lr_PLA', type=float, default=0.01, help='learning rate for PLA')
    parser.add_argument('--epsilon', type=float, default=0.01, help='error threshold for the capacity')
    parser.add_argument('--act_prob_range', type=float, nargs=2, default=[0.01, 0.4-0.01], help='range of the activation probability')
    parser.add_argument('--conn_prob_range', type=float, nargs=2, default=[0.01, 1-0.01], help='range of the connection probability')
    parser.add_argument('--distribution', type=str, default='uniform', help='distribution of the activation and connection probability',
    choices=['uniform', 'normal'])
    parser.add_argument('--change_in_degree', action='store_true', help='whether the in-degree connection probability is changed')
    parser.add_argument('--change_out_degree', action='store_true', help='whether the out-degree connection probability is changed')
    parser.add_argument('--use_reg', action='store_true', help='whether to use kappa')
    parser.add_argument('--kappa', type=float, default=0.0, help='kappa for the network')
    parser.add_argument('--n_repeat', type=int, default=2, help='number of repeats for the capacity calculation')
    parser.add_argument('--n_process', type=int, default=10, help='number of processes for parallel computation')

    parser.add_argument('--use_committee_machine', action='store_true', help='whether to use committee machine as neuron model')
    parser.add_argument('--n_dentrates', type=int, default=3, help='number of dendrites in each neuron model')
    parser.add_argument('--lr_committee', type=float, default=0.05, help='learning rate for committee machine')
    parser.add_argument('--add_bias', action='store_true', help='whether to add bias in committee machine')


    args = parser.parse_args()
    pars = vars(args)

    random.seed(25) # 10
    torch.manual_seed(34) # 11


    pars['act_prob_range'] = [0.01, 1.0-0.01] #[0.01, 0.4-0.01]
    pars['conn_prob_range'] = [0.01, 1-0.01]
    if not pars["use_reg"]:
        pars["kappa"] = 0.0
    n_repeat = pars["n_repeat"]
    epsilon = pars["epsilon"]
    n_neuron = pars["n_neuron"]
    n_process = np.min([mp.cpu_count(), pars["n_process"]])


    # 
    # 2D heatmap plot of the capacity, x is changing the neuron_act_prop, y is changing the connection heterogeneity
    # 1. Theoretical plot 2. numerical simulation.

    ratio_conn_std_list = np.arange(0.0, 0.2, 0.02) # np.arange(0.0,0.2, 0.02)
    pattern_act_std_list = np.arange(0.0, 0.141, 0.02) # np.arange(0.0,0.4, 0.02)

    capacities = np.zeros((len(ratio_conn_std_list), len(pattern_act_std_list), n_repeat))
    capacities_theoretical = np.zeros((len(ratio_conn_std_list), len(pattern_act_std_list), n_repeat))
    capacities_matched = np.zeros((len(ratio_conn_std_list), len(pattern_act_std_list), n_repeat))
    capacities_theoretical_matched = np.zeros((len(ratio_conn_std_list), len(pattern_act_std_list), n_repeat))
    capacities_anti_matched = np.zeros((len(ratio_conn_std_list), len(pattern_act_std_list), n_repeat))
    capacities_theoretical_anti_matched = np.zeros((len(ratio_conn_std_list), len(pattern_act_std_list), n_repeat))

    seed1_pool = np.random.randint(100000, size=(len(ratio_conn_std_list), len(pattern_act_std_list), n_repeat))
    seed2_pool = np.random.randint(100000, size=(len(ratio_conn_std_list), len(pattern_act_std_list), n_repeat))

    pool_results_object = [[[] for i in range(len(pattern_act_std_list))] for j in range(len(ratio_conn_std_list))]
    pool_results = [[[] for i in range(len(pattern_act_std_list))] for j in range(len(ratio_conn_std_list))]

    print(f"Start parallel computation, with {n_process} processes")
    t_start = time.time()
    
    with mp.Pool(processes=n_process) as pool:
        for i in range(len(ratio_conn_std_list)):
            for j in range(len(pattern_act_std_list)):
                for k in range(n_repeat):
                    print("ratio_conn_std: ", ratio_conn_std_list[i], "pattern_act_std: ", pattern_act_std_list[j])
                    # numerical capacity
                    pars_copy = pars.copy()
                    pars_copy["ratio_conn_std"] = ratio_conn_std_list[i]
                    pars_copy["pattern_act_std"] = pattern_act_std_list[j]

                    # pool_results_object[i][j].append(cacpacity_for_parallel(pars_copy, seed1_pool[k], seed2_pool[k]))
                    pool_results_object[i][j].append(pool.apply_async(cacpacity_for_parallel, args=(pars_copy, seed1_pool[i,j,k], seed2_pool[i,j,k])))
                    # for debug: use a for loop instead of parallel method to caculate the capacity:
                    # pool_results[i][j].append(cacpacity_for_parallel(pars_copy, seed1_pool[i,j,k], seed2_pool[i,j,k]))


        # get and pool the parallel results:

        for i in range(len(ratio_conn_std_list)):
            for j in range(len(pattern_act_std_list)):
                try:
                    pool_results[i][j] = [p.get() for p in pool_results_object[i][j]]
                except:
                    raise ValueError("Error in getting the results of pool_results[%d][%d]" % (i,j))
                # For debug: check if there is any error:
                if len(pool_results[i][j]) != n_repeat or pool_results[i][j][0] is None:
                    print("Error in pool_results[%d][%d]" % (i,j))
                
                capacities[i,j,:] = [pool_results[i][j][k][0] for k in range(n_repeat)]
                capacities_theoretical[i,j,:] = [pool_results[i][j][k][1] for k in range(n_repeat)]
                capacities_matched[i,j,:] = [pool_results[i][j][k][2] for k in range(n_repeat)]
                capacities_theoretical_matched[i,j,:] = [pool_results[i][j][k][3] for k in range(n_repeat)]
                capacities_anti_matched[i,j,:] = [pool_results[i][j][k][4] for k in range(n_repeat)]
                capacities_theoretical_anti_matched[i,j,:] = [pool_results[i][j][k][5] for k in range(n_repeat)]

    pool.close()
    pool.join()

    t_end = time.time()
    print(f"Parallel computation finished, time used: {t_end-t_start} seconds")


    # save the results:
    with open('results/Heterogeneous_committee_machine_network_capacity_k=%d_input_heter_=%d' % (pars["kappa"], pars["change_in_degree"]) + utils.make_name_with_time() + '.pkl', 'wb') as f:
        pickle.dump([capacities, capacities_theoretical, capacities_matched, capacities_theoretical_matched, capacities_anti_matched, capacities_theoretical_anti_matched, ratio_conn_std_list, pattern_act_std_list, pars], f)

    # make the colorbar scale the same for all subplots:
    vmin=0.0
    vmax = 0.91
    step = 0.05


    # plot the capacity: # TODO: refactor the code such that all plotting code is in one place.
    fig, axes = plt.subplots(3,3, figsize = [14,10])
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.4, hspace=0.4)

    capacities_mean = np.mean(capacities, axis=2)
    plt.sca(axes[0,0])
    # plot the meshgrid of capacity matrix:
    xx, yy = np.meshgrid(pattern_act_std_list, ratio_conn_std_list)
    h = plt.contourf(xx, yy, capacities_mean, cmap='viridis', vmin=vmin, vmax=vmax,levels = np.arange(vmin, vmax, step))
    # label the colorbar:
    cbar = plt.colorbar()
    cbar.set_label("Capacity", rotation=270, labelpad=20)
    plt.xlabel("Neuron activation heterogeneity")
    plt.ylabel("Connection heterogeneity")
    plt.title("Capacity of the network (Numerical simulation)")
    plt.axis('auto')

    # Theoretical estimation of the capacity:
    capacities_theoretical_mean = np.mean(capacities_theoretical, axis=2)
    plt.sca(axes[0,1])
    # plot the meshgrid of capacity matrix:
    xx, yy = np.meshgrid(pattern_act_std_list, ratio_conn_std_list)
    h = plt.contourf(xx, yy, capacities_theoretical_mean, cmap='viridis', vmin=vmin, vmax=vmax,levels = np.arange(vmin, vmax, step))
    # label the colorbar:
    cbar = plt.colorbar()
    cbar.set_label("Capacity", rotation=270, labelpad=20)
    plt.xlabel("Neuron activation heterogeneity")
    plt.ylabel("Connection heterogeneity")
    plt.title("Capacity of the network (Theoretical estimation)")
    plt.axis('auto')

    capacities_matched_mean = np.mean(capacities_matched, axis=2)
    plt.sca(axes[1,0])
    # plot the meshgrid of capacity matrix:
    xx, yy = np.meshgrid(pattern_act_std_list, ratio_conn_std_list)
    h = plt.contourf(xx, yy, capacities_matched_mean, cmap='viridis', vmin=vmin, vmax=vmax,levels = np.arange(vmin, vmax, step))
    # label the colorbar:
    cbar = plt.colorbar()
    cbar.set_label("Capacity", rotation=270, labelpad=20)
    plt.xlabel("Neuron activation heterogeneity")
    plt.ylabel("Connection heterogeneity")
    plt.title("Connection and activation positive correlated (Numerical simulation)")
    plt.axis('auto')

    capacities_theoretical_matched_mean = np.mean(capacities_theoretical_matched, axis=2)
    plt.sca(axes[1,1])
    # plot the meshgrid of capacity matrix:
    xx, yy = np.meshgrid(pattern_act_std_list, ratio_conn_std_list)
    h = plt.contourf(xx, yy, capacities_theoretical_matched_mean, cmap='viridis', vmin=vmin, vmax=vmax,levels = np.arange(vmin, vmax, step))
    # label the colorbar:
    cbar = plt.colorbar()
    cbar.set_label("Capacity", rotation=270, labelpad=20)
    plt.xlabel("Neuron activation heterogeneity")
    plt.ylabel("Connection heterogeneity")
    plt.title("Connection and activation positive correlated (Theoretical estimation)")
    plt.axis('auto')

    capacities_anti_matched_mean = np.mean(capacities_anti_matched, axis=2)
    plt.sca(axes[2,0])
    # plot the meshgrid of capacity matrix:
    xx, yy = np.meshgrid(pattern_act_std_list, ratio_conn_std_list)
    h = plt.contourf(xx, yy, capacities_anti_matched_mean, cmap='viridis', vmin=vmin, vmax=vmax,levels = np.arange(vmin, vmax, step))
    # label the colorbar:
    cbar = plt.colorbar()
    cbar.set_label("Capacity", rotation=270, labelpad=20)
    plt.xlabel("Neuron activation heterogeneity")
    plt.ylabel("Connection heterogeneity")
    plt.title("Connection and activation negative correlated (Numerical simulation)")
    plt.axis('auto')

    capacities_theoretical_anti_matched_mean = np.mean(capacities_theoretical_anti_matched, axis=2)
    plt.sca(axes[2,1])
    # plot the meshgrid of capacity matrix:
    xx, yy = np.meshgrid(pattern_act_std_list, ratio_conn_std_list)
    h = plt.contourf(xx, yy, capacities_theoretical_anti_matched_mean, cmap='viridis', vmin=vmin, vmax=vmax,levels = np.arange(vmin, vmax, step))
    # label the colorbar:
    cbar = plt.colorbar()
    cbar.set_label("Capacity", rotation=270, labelpad=20)
    plt.xlabel("Neuron activation heterogeneity")
    plt.ylabel("Connection heterogeneity")
    plt.title("Connection and activation negative correlated (Theoretical estimation)")
    plt.axis('auto')

    utils.print_parameters_on_plot(axes[2,2], pars)
    plt.show(block=False)

    # plt.savefig('results/Heterogeneous_network_capcity'+utils.make_name_with_time()+'.jpg')
    plt.savefig('results/Heterogeneous_committee_machine_network_capcity'+utils.make_name_with_time()+'.pdf')

    print("end")

