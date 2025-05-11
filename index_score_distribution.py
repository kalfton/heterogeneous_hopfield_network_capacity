import numpy as np
from numpy import random
import torch
from matplotlib import pyplot as plt
from random_connection_network import random_connect_network
import utils_basic as utils
import warnings
import utils_numerical as utils_num
import argparse
import sys
from theoretical_two_layer_network import theoretical_two_layer_network_capacity

if __name__ == "__main__":
    # sys.argv += "--n_neuron 150 --n_neuron_L2 450 --network_type RBM --method svm".split()

    parser = argparse.ArgumentParser(description='Train neural dependency parser in pytorch')
    parser.add_argument('--method', type=str, default='hebbian', help='training method',
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
    parser.add_argument('--neuron_base_state', type=float, default=-1, help='the base state of the neurons')
    parser.add_argument('--kappa', type=float, default=0.0, help='the kappa value for the theoretical capacity')
    parser.add_argument('--use_reg', action='store_true', help='whether to use regularization')

    args = parser.parse_args()
    pars = vars(args)

    if not pars['use_reg']:
        pars['kappa'] = 0.0


    n_neuron = pars["n_neuron"]
    n_neuron_L2 =  pars["n_neuron_L2"]
    training_max_iter = pars["training_max_iter"]
    weight_std_init = pars["weight_std_init"]
    method = pars["method"]
    lr_PLA = pars["lr_PLA"]
    W_symmetric = pars["W_symmetric"]
    network_type = pars["network_type"]
    neuron_base_state = pars["neuron_base_state"]
    use_reg = pars["use_reg"]
    kappa = pars["kappa"]

    random.seed(pars['seed1'])
    torch.manual_seed(pars['seed2'])

    perc_act_L1_set = np.concatenate((np.arange(0.01, 0.0999, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.9999, 0.03)))
    perc_act_L2_set = np.concatenate((np.arange(0.01, 0.0999, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.9999, 0.03)))

    Capacity, information_capcity_per_neuron_square, _, _ = theoretical_two_layer_network_capacity(perc_act_L1_set, perc_act_L2_set, n_neuron, n_neuron_L2, use_reg, kappa, make_plot=False)

    n_pattern_1 = int(Capacity[7,7]*(n_neuron+n_neuron_L2))
    n_pattern_2 = int(Capacity[0,3]*(n_neuron+n_neuron_L2))
    n_pattern_3 = int(Capacity[0,7]*(n_neuron+n_neuron_L2))

    # Choose three points in the coding level space to plot the distribution of the Index scores:
    parameter_sets = [
        {"perc_active_L1": 0.5, "perc_active_L2": 0.5,  "n_pattern": n_pattern_1},
        {"perc_active_L1": 0.1, "perc_active_L2": 0.01, "n_pattern": n_pattern_2},
        {"perc_active_L1": 0.5, "perc_active_L2": 0.01, "n_pattern": n_pattern_3}
    ]
    colorsets = ['y', 'r', 'b']

    statistics = []

    fig, axes = plt.subplots(3,3, figsize = [19,10])
    for paras in parameter_sets:
        
        perc_active_L1 = paras["perc_active_L1"]
        perc_active_L2 = paras["perc_active_L2"]
        n_pattern = paras["n_pattern"]

        patterns_L1 = utils.make_pattern(n_pattern, n_neuron, perc_active=perc_active_L1, neuron_base_state=neuron_base_state)
        patterns_L2 = utils.make_pattern(n_pattern, n_neuron_L2, perc_active=perc_active_L2, neuron_base_state=neuron_base_state)

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
        unique_patterns_all = torch.unique(patterns_all, dim=0)

        if method.lower() == 'PLA'.lower():
            success, stored_patterns_all = network.train_PLA(patterns_all, training_max_iter=training_max_iter, lr=lr_PLA)
        elif method.lower() == 'svm'.lower():
            success, stored_patterns_all = network.train_svm(patterns_all, use_reg = pars["use_reg"], kappa = pars["kappa"])

        stored_patterns_L1 = stored_patterns_all[:,:n_neuron]
        stored_patterns_L2 = stored_patterns_all[:,n_neuron:]


        W_21_trained = network.weight.clone()[n_neuron:, :n_neuron]
        W_12_trained = network.weight.clone()[:n_neuron, n_neuron:]
        b_1_trained = network.b.clone()[:n_neuron]
        b_2_trained = network.b.clone()[n_neuron:]

        bins = np.arange(0,1.01,0.01)
        utils_num.index_analysis_v4(W_12_trained, W_21_trained.mT, patterns_L1=patterns_L1, patterns_L2= patterns_L2, perc_active_L1=perc_active_L1, \
                                                neuron_base_state = neuron_base_state, ax = axes[0,0], bins=bins, color=colorsets[parameter_sets.index(paras)])
        
        plt.xlim([0.25,1])
        plt.xticks(np.arange(0.25,1.1,0.25))

    utils.print_parameters_on_plot(axes[2,2], pars)
    # plt.show(block=False)
    plt.savefig('results/index_analysis_combined_hist'+utils.make_name_with_time()+'.pdf')