import numpy as np
from numpy import random
import torch
from matplotlib import pyplot as plt
import utils_basic as utils
import pickle
import warnings
import time
import argparse
import sys
import multiprocessing as mp

from numerical_two_layer_network_parallel import calculate_numerical_capacity

        
if __name__ == '__main__':

    # sys.argv += "--network_type RBM --method svm --training_max_iter 1000 \
    #     --neuron_base_state -1 --epsilon 0.01 --n_repeat 50 --n_process 10".split() #--use_reg --kappa 0.5
    
    parser = argparse.ArgumentParser(description='Two layer network finite size analysis')
    parser.add_argument('--method', type=str, default='svm', help='training method',
    choices=['PLA', 'svm'])
    parser.add_argument('--W_symmetric', action='store_true', help='whether the W matrix is symmetric')
    parser.add_argument('--training_max_iter', type=int, default=10000, help='maximum number of iterations for training')
    parser.add_argument('--weight_std_init', type=float, default=0.1, help='standard deviation of the initial weights')
    parser.add_argument('--lr_PLA', type=float, default=0.02, help='learning rate for PLA')
    parser.add_argument('--seed1', type=int, default=20, help='seed for random number generator')
    parser.add_argument('--seed2', type=int, default=143, help='seed for random number generator')
    parser.add_argument('--network_type', type=str, default='RBM', help='type of network')
    parser.add_argument('--n_repeat', type=int, default=10, help='number of times to replicate the experiment')
    parser.add_argument('--neuron_base_state', type=float, default=-1, help='the base state of the neurons')
    parser.add_argument('--epsilon', type=float, default=0.01, help='the error threshold for the capacity')
    parser.add_argument('--use_reg', action='store_true', help='whether to use kappa')
    parser.add_argument('--kappa', type=float, default=0.0, help='the kappa for the training')
    parser.add_argument('--n_process', type=int, default=10, help='number of processes to run in parallel')



    args = parser.parse_args()
    pars = vars(args)

    training_max_iter = pars["training_max_iter"]
    weight_std_init = pars["weight_std_init"]
    method = pars["method"]
    lr_PLA = pars["lr_PLA"]
    W_symmetric = pars["W_symmetric"]
    network_type = pars["network_type"]
    n_repeat = pars["n_repeat"]
    neuron_base_state = pars["neuron_base_state"]
    epsilon = pars["epsilon"]
    use_reg = pars["use_reg"]

    if use_reg:
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
    perc_act_L1_set = np.arange(0.01, 0.52, 0.02)
    perc_act_L2_set = np.arange(0.01, 0.21, 0.01)

    seed1_pool_all = np.random.randint(100000, size=(len(perc_act_L2_set), len(perc_act_L1_set), repeating_measure))
    seed2_pool_all = np.random.randint(100000, size=(len(perc_act_L2_set), len(perc_act_L1_set), repeating_measure))

    n_neuron_L1_set = np.array([10, 15, 25, 50, 75])
    n_neuron_L2_set = 3*n_neuron_L1_set

    arg_max_perc_act_L1 = np.zeros(len(n_neuron_L1_set))
    arg_max_perc_act_L2 = np.zeros(len(n_neuron_L1_set))
    max_capacities = np.zeros(len(n_neuron_L1_set))
    capacities_matrix = np.zeros((len(n_neuron_L1_set), len(perc_act_L2_set), len(perc_act_L1_set)))
    for iii in range(len(n_neuron_L1_set)):
        n_neuron = n_neuron_L1_set[iii]
        n_neuron_L2 = n_neuron_L2_set[iii]
        pars["n_neuron"] = n_neuron
        pars["n_neuron_L2"] = n_neuron_L2
        print("n_neuron = %d" % n_neuron)
        print("n_neuron_L2 = %d" % n_neuron_L2)

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
                    pool_results_object[i][j] = [pool.apply_async(calculate_numerical_capacity, args=(pars_copy, n_pattern_max, step, epsilon, seed1_pool[k], seed2_pool[k]) ) for k in range(repeating_measure)]


            # get the parallel results:
            for i in range(len(perc_act_L2_set)):
                for j in range(len(perc_act_L1_set)):
                    try:
                        pool_results[i][j] = [p.get() for p in pool_results_object[i][j]]
                        capacities[i][j] = np.mean(pool_results[i][j])
                    except:
                        raise ValueError("Error in getting the results of pool_results[%d][%d]" % (i,j))
                    # Check if there is any error:
                    if len(pool_results[i][j]) != repeating_measure or pool_results[i][j][0] is None or pool_results[i][j][1] is None:
                        print("Error in pool_results[%d][%d]" % (i,j))
                        print("length of pool_results[%d][%d] = %d" % (i,j, len(pool_results[i][j])))
                        print("perc_active_1 = %f" % perc_act_L1[j])
                        print("perc_active_2 = %f" % perc_act_L2[i])
                        print("seed1_pool = {}".format(seed1_pool_all[i,j]))
                        print("seed2_pool = {}".format(seed2_pool_all[i,j]))

        pool.close()
        pool.join()
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

        plt.savefig('results/numerical_two_layer_capacity_and_information_capacity_finite_size_'+utils.make_name_with_time()+'.pdf')

        # # save the results as pickle file:
        with open('results/numerical_two_layer_finite_size_'+'k = %.1f, n_layer1 = %d, n_layer_2 = %d, repeat = %d' % (kappa, n_neuron, n_neuron_L2, n_repeat) +'.pkl', 'wb') as f:
            pickle.dump([capacities, information_capacity, perc_act_L1_set, perc_act_L2_set ,pars], f)
        print("end")