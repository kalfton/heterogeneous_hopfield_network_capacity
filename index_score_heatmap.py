import numpy as np
import torch
from matplotlib import pyplot as plt
from random_connection_network import random_connect_network
import utils_basic as utils
import utils_numerical as utils_num
import pickle
import warnings
import time
import argparse
import sys
from theoretical_two_layer_network import theoretical_two_layer_network_capacity

if __name__ == "__main__":
    # sys.argv += "--n_neuron 50 --n_neuron_L2 150 --network_type RBM --method svm --training_max_iter 1000 --n_repeat 2".split() #--use_reg --kappa 0.5

    parser = argparse.ArgumentParser(description='Train neural dependency parser in pytorch')
    parser.add_argument('--method', type=str, default='svm', help='training method',
    choices=['PLA', 'svm'])
    parser.add_argument('--n_neuron', type=int, default=100, help='number of neurons in the first layer')
    parser.add_argument('--n_neuron_L2', type=int, default=100, help='number of neurons in the second layer')
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
    parser.add_argument('--use_reg', action='store_true', help='whether to use regularization with finite kappa')
    parser.add_argument('--kappa', type=float, default=0.0, help='the kappa for the training')
    parser.add_argument('--n_process', type=int, default=10, help='number of processes to run in parallel')

    args = parser.parse_args()
    pars = vars(args)

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
    use_reg = pars["use_reg"]
    kappa = pars["kappa"]

    perc_act_L1_set = np.concatenate((np.arange(0.01, 0.0999, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.9999, 0.03)))
    perc_act_L2_set = np.concatenate((np.arange(0.01, 0.0999, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.9999, 0.03)))


    # get the theoretical capacity values:
    capacities, information_capacity, _, _ = theoretical_two_layer_network_capacity(perc_act_L1_set, perc_act_L2_set, n_neuron, n_neuron_L2, use_reg, kappa, make_plot=False)
    # unnormalize the capacity values:
    capacities = (np.floor(capacities*(n_neuron+n_neuron_L2))).astype(int)
    information_capacity = information_capacity*(n_neuron+n_neuron_L2)**2


    hipp_index_score_1 = np.zeros((len(perc_act_L2_set), len(perc_act_L1_set)))
    for i in range(len(perc_act_L2_set)):
        for j in range(len(perc_act_L1_set)):
            print("perc_act_L1 = %f, perc_act_L2 = %f, capacity = %f" % (perc_act_L1_set[j], perc_act_L2_set[i], capacities[i][j]))
            n_pattern = int(np.floor(capacities[i][j]))
            if n_pattern <=0: #2
                hipp_index_score_1[i,j] = np.nan
                continue
            print("n_pattern = %d" % n_pattern)
            # make the patterns:
            patterns_L1 = utils.make_pattern(n_pattern, n_neuron, perc_active=perc_act_L1_set[j], neuron_base_state=neuron_base_state)
            patterns_L2 = utils.make_pattern(n_pattern, n_neuron_L2, perc_active=perc_act_L2_set[i], neuron_base_state=neuron_base_state)

            # initialize the network:
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
                if pars["use_reg"]:
                    success, stored_patterns_all = network.train_svm(patterns_all, use_reg = True, kappa = pars["kappa"])
                else:
                    success, stored_patterns_all = network.train_svm(patterns_all, use_reg = False)


            # index theory analysis:
            W_12_trained = network.weight.clone().numpy()[:n_neuron, n_neuron:]
            W_21_trained = network.weight.clone().numpy()[n_neuron:, :n_neuron]

            index_score = utils_num.index_analysis_v4(W_12_trained, W_21_trained.T, patterns_L1=patterns_L1,patterns_L2=patterns_L2, perc_active_L1=perc_act_L1_set[j], neuron_base_state = neuron_base_state, makeplot=False)
            hipp_index_score_1[i,j] = index_score
            

    with open('results/numerical_two_layer_hippo_index_L1'+str(n_neuron)+'_L2'+str(n_neuron_L2)+'_kappa'+str(pars["kappa"])+'.pkl', 'wb') as f:
        pickle.dump([hipp_index_score_1, pars], f)

    # plot the index score:
    fig, axes = plt.subplots(2,2, figsize = [19,10])

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.4, hspace=0.4)


    utils_num.heatmap_plot(perc_act_L1_set, perc_act_L2_set, hipp_index_score_1, x_label="Activation probability in layer 1", y_label="Activation probability in layer 2",\
                        z_label="Network index score", title = "Index score", step = 0.005, ax=axes[0,0])


    utils.print_parameters_on_plot(axes[1,1], pars)
    plt.show(block=False)
    plt.savefig('results/numerical_two_layer_hippo_index' +utils.make_name_with_time()+'.pdf')
