import numpy as np
from numpy import random
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import scipy
from random_connection_network import random_connect_network
import utils_basic as utils
# import utils_numerical as utils_num
import pickle
import warnings
import time
import argparse
import sys

import multiprocessing as mp

def caculate_numerical_cacpacity(pars, max_n_pattern:int, step:int, epsilon, seed1, seed2):
    n_pattern = pars["n_pattern"]
    n_neuron = pars["n_neuron"]
    n_neuron_L2 =  pars["n_neuron_L2"]
    perc_active_L1 = pars["perc_active_L1"]
    perc_active_L2 = pars["perc_active_L2"]
    evolve_step = pars["evolve_step"]
    dt_train= pars["dt_train"]
    training_max_iter = pars["training_max_iter"]
    weight_std_init = pars["weight_std_init"]
    method = pars["method"]
    lr_adam = pars["lr_adam"]
    lr_SGD = pars["lr_SGD"]
    lr_PLA = pars["lr_PLA"]
    gamma_SGD = pars["gamma_SGD"]
    W_symmetric = not pars["W_notsymmetric"]
    network_type = pars["network_type"]
    neuron_base_state = pars["neuron_base_state"]
    use_kappa = pars["use_kappa"]
    kappa = pars["kappa"]

    random.seed(seed1)
    torch.manual_seed(seed2)
    # Binary search:
    # there is not enough patterns to be stored: lower the max_n_pattern
    patterns_L1 = utils.make_pattern(max_n_pattern, n_neuron, perc_active=perc_active_L1, neuron_base_state=neuron_base_state)
    patterns_L2 = utils.make_pattern(max_n_pattern, n_neuron_L2, perc_active=perc_active_L2, neuron_base_state=neuron_base_state)
    # eliminate the repeating patterns:
    patterns_all = torch.unique(torch.cat((patterns_L1, patterns_L2),dim=1), dim=0)
    n_pattern_unique = patterns_all.shape[0]
    # n_pattern_unique_L1 = torch.unique(patterns_L1, dim=0).shape[0]
    # n_pattern_unique_L2 = torch.unique(patterns_L2, dim=0).shape[0]
    # n_pattern_unique = min(n_pattern_unique, n_pattern_unique_L1, n_pattern_unique_L2)
    left, right = 1, n_pattern_unique
    
    while left <= right:
        # make the binary search uneven, because when it is under capacity, the calculation is faster:
        mid = (left + right) // 2 #int(0.75*left+0.25*right)  #
        n_pattern = mid
        patterns_L1 = utils.make_pattern(n_pattern, n_neuron, perc_active=perc_active_L1, neuron_base_state=neuron_base_state)
        patterns_L2 = utils.make_pattern(n_pattern, n_neuron_L2, perc_active=perc_active_L2, neuron_base_state=neuron_base_state)
        # eliminate the repeating patterns:
        # patterns_all = torch.unique(torch.cat((patterns_L1, patterns_L2),dim=1), dim=0)
        # n_pattern_unique = patterns_all.shape[0]
        # mid = n_pattern_unique
        if network_type.lower() == 'RBM'.lower():
            mask = torch.ones((n_neuron+n_neuron_L2, n_neuron+n_neuron_L2))
            mask[:n_neuron, :n_neuron] = 0
            mask[n_neuron:, n_neuron:] = 0
        elif network_type.lower() == 'Hybrid'.lower():
            mask = torch.ones((n_neuron+n_neuron_L2, n_neuron+n_neuron_L2))
            mask-=torch.eye(n_neuron+n_neuron_L2)
            mask[n_neuron:, n_neuron:] = 0
            mask.fill_diagonal_(0)
        elif network_type.lower() == 'Full'.lower():
            mask = torch.ones((n_neuron+n_neuron_L2, n_neuron+n_neuron_L2))
            mask.fill_diagonal_(0)
        else:
            raise ValueError("network_type should be 'RBM', 'Hybrid', or 'Full'.")
        
        network = random_connect_network(n_neuron+n_neuron_L2, W_symmetric=W_symmetric,connection_prop=None, weight_std_init=weight_std_init, mask=mask, neuron_base_state=neuron_base_state)
        patterns_all = torch.cat((patterns_L1, patterns_L2), dim=1)
        if method.lower() == 'PLA'.lower():
            success, stored_patterns_all = network.train_PLA(patterns_all, training_max_iter=training_max_iter, lr=lr_PLA)
        elif method.lower() == 'svm'.lower():
            if use_kappa:
                success, stored_patterns_all = network.train_svm(patterns_all, kappa=kappa)
            else:
                success, stored_patterns_all = network.train_svm(patterns_all, use_reg = False)

        mean_error = 1-utils.network_score(network.weight, network.b, patterns_all, kappa = kappa)

        
        if mean_error>epsilon: # the capacity is smaller than n_pattern
            right = mid - step
        else: # the capacity is larger than n_pattern
            left = mid + step

    return left


        
if __name__ == '__main__':

    sys.argv += "--W_notsymmetric --n_neuron 150 --n_neuron_L2 450 --network_type RBM --method svm --training_max_iter 1000 --neuron_base_state -1 --epsilon 0.01 --n_repeat 2".split() #--use_kappa --kappa 0.5
    
    parser = argparse.ArgumentParser(description='Train neural dependency parser in pytorch')
    parser.add_argument('--method', type=str, default='svm', help='training method',
    choices=['Adam', 'SGD', 'PLA', 'svm'])
    parser.add_argument('--n_pattern', type=int, default=20, help='number of patterns to store')
    parser.add_argument('--n_neuron', type=int, default=100, help='number of neurons in the first layer')
    parser.add_argument('--n_neuron_L2', type=int, default=100, help='number of neurons in the second layer')
    parser.add_argument('--perc_active_L1', type=float, default=0.5, help='percentage of active neurons in the first layer')
    parser.add_argument('--perc_active_L2', type=float, default=0.5, help='percentage of active neurons in the second layer')
    parser.add_argument('--W_notsymmetric', action='store_true', help='whether the W matrix is symmetric')
    parser.add_argument('--evolve_step', type=int, default=50, help='number of steps to evolve the network') # 50 for eq_prop and 10 for back_prop
    parser.add_argument('--dt_train', type=float, default=0.1, help='time step for training')
    parser.add_argument('--training_max_iter', type=int, default=10000, help='maximum number of iterations for training')
    parser.add_argument('--weight_std_init', type=float, default=0.1, help='standard deviation of the initial weights')
    parser.add_argument('--lr_adam', type=float, default=0.01, help='learning rate for Adam')
    parser.add_argument('--lr_SGD', type=float, default=10, help='learning rate for SGD')
    parser.add_argument('--lr_PLA', type=float, default=0.02, help='learning rate for PLA')
    parser.add_argument('--gamma_SGD', type=float, default=0.9999, help='gamma for SGD')
    parser.add_argument('--seed1', type=int, default=20, help='seed for random number generator')
    parser.add_argument('--seed2', type=int, default=143, help='seed for random number generator')
    parser.add_argument('--network_type', type=str, default='RBM', help='type of network')
    parser.add_argument('--n_repeat', type=int, default=10, help='number of times to replicate the experiment')
    parser.add_argument('--neuron_base_state', type=float, default=-1, help='the base state of the neurons')
    parser.add_argument('--epsilon', type=float, default=0.01, help='the error threshold for the capacity')
    parser.add_argument('--use_kappa', action='store_true', help='whether to use kappa')
    parser.add_argument('--kappa', type=float, default=0.0, help='the kappa for the training')
    parser.add_argument('--n_process', type=int, default=10, help='number of processes to run in parallel')



    args = parser.parse_args()
    pars = vars(args)


    n_pattern = pars["n_pattern"]
    n_neuron = pars["n_neuron"]
    n_neuron_L2 =  pars["n_neuron_L2"]
    perc_active_L1 = pars["perc_active_L1"]
    perc_active_L2 = pars["perc_active_L2"]
    evolve_step = pars["evolve_step"]
    dt_train= pars["dt_train"]
    training_max_iter = pars["training_max_iter"]
    weight_std_init = pars["weight_std_init"]
    method = pars["method"]
    lr_adam = pars["lr_adam"]
    lr_SGD = pars["lr_SGD"]
    lr_PLA = pars["lr_PLA"]
    gamma_SGD = pars["gamma_SGD"]
    W_symmetric = not pars["W_notsymmetric"]
    network_type = pars["network_type"]
    n_repeat = pars["n_repeat"]
    neuron_base_state = pars["neuron_base_state"]
    epsilon = pars["epsilon"]
    use_kappa = pars["use_kappa"]

    if use_kappa:
        kappa = pars["kappa"]
    else:
        pars["kappa"] = 0.0
        kappa = 0.0

    n_process = np.min([mp.cpu_count(), pars["n_process"]])
    print("n_process = %d" % n_process)

    random.seed(pars['seed1'])
    torch.manual_seed(pars['seed2'])

    # numerically calculate the capacity of the network:
    
    repeating_measure = n_repeat
    # n_act_L2_set = np.concatenate((np.arange(1,5), np.arange(6, n_neuron_L2, 3)))
    perc_act_L1_set = np.concatenate((np.arange(0.01, 0.0999, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.9999, 0.03))) #
    perc_act_L2_set = np.concatenate((np.arange(0.01, 0.0999, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.9999, 0.03)))

    seed1_pool_all = np.random.randint(100000, size=(len(perc_act_L2_set), len(perc_act_L1_set), repeating_measure))
    seed2_pool_all = np.random.randint(100000, size=(len(perc_act_L2_set), len(perc_act_L1_set), repeating_measure))

    capacities = np.zeros((len(perc_act_L2_set), len(perc_act_L1_set)))
    pool_results_object = [[[] for i in range(len(perc_act_L1_set))] for j in range(len(perc_act_L2_set))]
    pool_results = [[[] for i in range(len(perc_act_L1_set))] for j in range(len(perc_act_L2_set))]
    step = int(max(n_neuron, n_neuron_L2)/10)
    n_pattern_max = 7*(max(n_neuron, n_neuron_L2))
    
    print("n_neuron = %d" % n_neuron)
    print("n_neuron_L2 = %d" % n_neuron_L2)

    start_time = time.time()
    with mp.Pool(processes=n_process) as pool:
        for i in range(len(perc_act_L2_set)):
            perc_act_L2 = perc_act_L2_set[i]
            for j in range(len(perc_act_L1_set)):
                perc_act_L1 = perc_act_L1_set[j]

                # make it run in parallel:
                seed1_pool = seed1_pool_all[i,j,:] #np.random.randint(100000, size=repeating_measure)
                seed2_pool = seed2_pool_all[i,j,:] #np.random.randint(100000, size=repeating_measure)
                pars["perc_active_L1"] = perc_act_L1
                pars["perc_active_L2"] = perc_act_L2
                
                print("perc_active_1 = %f" % perc_act_L1)
                print("perc_active_2 = %f" % perc_act_L2)
                print("seed1_pool = {}".format(seed1_pool[0]))
                print("seed2_pool = {}".format(seed2_pool[0]))
                # run the parallel processing:
                pars_copy = pars.copy()
                pool_results_object[i][j] = [pool.apply_async(caculate_numerical_cacpacity, args=(pars_copy, n_pattern_max, step, epsilon, seed1_pool[k], seed2_pool[k]) ) for k in range(repeating_measure)]

                # for loop for debugging:
                # cap = np.zeros(repeating_measure)
                # for i_repeat in range(repeating_measure):
                #     cap[i_repeat] = caculate_numerical_cacpacity(pars_copy, n_pattern_max, step, epsilon, seed1_pool[i_repeat], seed2_pool[i_repeat])
                # pool_results[i][j] = cap


        # get and pool the parallel results:

        for i in range(len(perc_act_L2_set)):
            for j in range(len(perc_act_L1_set)):
                try:
                    pool_results[i][j] = [p.get() for p in pool_results_object[i][j]]
                except:
                    raise ValueError("Error in getting the results of pool_results[%d][%d]" % (i,j))
                # For debug: check if there is any error:
                if len(pool_results[i][j]) != repeating_measure or pool_results[i][j][0] is None or pool_results[i][j][1] is None:
                    print("Error in pool_results[%d][%d]" % (i,j))
                    print("length of pool_results[%d][%d] = %d" % (i,j, len(pool_results[i][j])))
                    print("perc_active_1 = %f" % perc_act_L1[j])
                    print("perc_active_2 = %f" % perc_act_L2[i])
                    print("seed1_pool = {}".format(seed1_pool_all[i,j]))
                    print("seed2_pool = {}".format(seed2_pool_all[i,j]))

    pool.close()
    pool.join()
        
    
    for i in range(len(perc_act_L2_set)):
        for j in range(len(perc_act_L1_set)):
            try:
                capacities[i][j] = np.mean(pool_results[i][j])
            except:
                print("Error in calculating the mean of pool_results[%d][%d]" % (i,j))
                print("pool_results[%d][%d] = %s" % (i,j, pool_results[i][j]))
                print("perc_active_1 = %f" % perc_act_L1[j])
                print("perc_active_2 = %f" % perc_act_L2[i])
                print("seed1_pool = {}".format(seed1_pool_all[i,j]))
                print("seed2_pool = {}".format(seed2_pool_all[i,j]))
                raise ValueError("Error in calculating the mean of pool_results[%d][%d]" % (i,j))
                    

        

    print("--- %s seconds ---" % (time.time() - start_time))

    # information entropy capacity:
    information_capacity = np.zeros((len(perc_act_L2_set), len(perc_act_L1_set)))
    for i in range(len(perc_act_L2_set)):
        for j in range(len(perc_act_L1_set)):
            information_capacity[i,j] = n_neuron*capacities[i][j]*utils.information_entropy(perc_act_L1_set[j])

    # plot the capacity:
    fig, axes = plt.subplots(2,2, figsize = [19,10])
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.4, hspace=0.4)
    plt.sca(axes[0,0])
    # plot the meshgrid of capacity matrix:
    frac_capacities = capacities/(n_neuron+n_neuron_L2)
    vmin=np.floor(np.min(frac_capacities)*10)/10-0.1
    vmin = max(vmin, 0.0)
    vmax=np.ceil(np.max(frac_capacities)*10)/10+0.1
    # vmax = 5.0
    # vmin = 0.0
    step = 0.1
    xx, yy = np.meshgrid(perc_act_L1_set, perc_act_L2_set)
    h = plt.contourf(xx, yy, frac_capacities, cmap='viridis', vmin=vmin, vmax=vmax, levels = np.arange(vmin, vmax+0.001, step), norm=utils.Rescaled_Norm(vmin = vmin, vmax = vmax))

    plt.axis('scaled')
    # label the colorbar:
    cbar = plt.colorbar()
    cbar.set_label("Capacity/neuron", rotation=270, labelpad=20)
    plt.xlabel("Activation probability in layer 1")
    plt.ylabel("Activation probabiltiy in layer 2")
    # make the axis square:
    plt.gca().set_aspect('equal', adjustable='box')

    # plot the information capacity
    plt.sca(axes[0,1])
    # plot the meshgrid of capacity matrix:
    info_capacities_per_neuron_square = information_capacity/(n_neuron+n_neuron_L2)**2 # information per neuron square
    # vmin=np.floor(np.min(info_capacities_per_neuron_square)*10)/10-0.1
    # vmin = max(vmin, 0.0)
    # vmax=np.ceil(np.max(info_capacities_per_neuron_square)*10)/10+0.1
    vmax = 0.5
    vmin = 0.0
    step = 0.02
    xx, yy = np.meshgrid(perc_act_L1_set, perc_act_L2_set)
    h = plt.contourf(xx, yy, info_capacities_per_neuron_square, cmap='viridis', vmin=vmin, vmax=vmax, levels = np.arange(vmin, vmax+0.001, step), norm=utils.Rescaled_Norm(vmin = vmin, vmax = vmax))

    plt.axis('scaled')
    # label the colorbar:
    cbar = plt.colorbar()
    cbar.set_label("Information capacity/neuron", rotation=270, labelpad=20)
    plt.xlabel("Activation probability in layer 1")
    plt.ylabel("Activation probabiltiy in layer 2")
    # make the axis square:
    plt.gca().set_aspect('equal', adjustable='box')

    utils.print_parameters_on_plot(axes[1,1], pars)
    plt.show(block=False)

    plt.savefig('results/numerical_two_layer_capacity_and_information_capacity'+utils.make_name_with_time()+'.pdf')
    
    # # save the results as pickle:
    with open('results/numerical_two_layer_temp_'+'k = %.1f, n_layer1 = %d, n_layer_2 = %d' % (kappa, n_neuron, n_neuron_L2) +'.pkl', 'wb') as f:
        pickle.dump([capacities, information_capacity, perc_act_L1_set, perc_act_L2_set ,pars], f)
    print("end")

