import numpy as np
from numpy import random
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import scipy
# from two_layer_hopfield_simplified import Two_layer_hopfield
from random_connection_network import random_connect_network
import utils_basic as utils
import pickle
import warnings
import utils_numerical as utils_num
import time
import argparse
import sys


sys.argv += "--W_notsymmetric --n_neuron 150 --n_neuron_L2 450 --n_pattern 500 --perc_active_L1 0.5 --perc_active_L2 0.01 \
    --network_type RBM --method svm --neuron_base_state -1 --kappa 0".split()

parser = argparse.ArgumentParser(description='Train neural dependency parser in pytorch')
parser.add_argument('--method', type=str, default='hebbian', help='training method',
choices=['Adam', 'SGD', 'PLA', 'svm', 'hebbian'])
parser.add_argument('--n_pattern', type=int, default=100, help='number of patterns to store')
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
parser.add_argument('--n_replicate', type=int, default=2, help='number of times to replicate the experiment')
parser.add_argument('--neuron_base_state', type=float, default=-1, help='the base state of the neurons')
parser.add_argument('--kappa', type=float, default=0.0, help='the kappa value for the theoretical capacity')
parser.add_argument('--use_reg', action='store_true', help='whether to use regularization')

args = parser.parse_args()
pars = vars(args)

if not pars['use_reg']:
    pars['kappa'] = 0.0


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
n_replicate = pars["n_replicate"]
neuron_base_state = pars["neuron_base_state"]

random.seed(pars['seed1'])
torch.manual_seed(pars['seed2'])



# read from the file:
if pars["use_reg"] and pars["kappa"] == 0.5:
    with open('results/theoretical_two_layer_capacity_L1_150_L2_450_k_0.5.pkl', 'rb') as f:
        Capacity, information_capcity_per_neuron_square, ratio_active_L1, ratio_active_L2 = pickle.load(f)
elif not pars["use_reg"] and n_neuron == 150 and n_neuron_L2 == 450:
    with open('results/theoretical_two_layer_capacity_L1_150_L2_450.pkl', 'rb') as f:
        Capacity, information_capcity_per_neuron_square, ratio_active_L1, ratio_active_L2 = pickle.load(f)
elif not pars["use_reg"] and n_neuron == 450 and n_neuron_L2 == 150:
    with open('results/theoretical_two_layer_capacity_L1_450_L2_150.pkl', 'rb') as f:
        Capacity, information_capcity_per_neuron_square, ratio_active_L1, ratio_active_L2 = pickle.load(f)

else:
    raise ValueError("The file does not exist")

# For kappa =0, the n_pattern respectively should be: 300, 1760, 900
# For kappa =0.5, the n_pattern respectively should be: 144, 882, 432

n_neuron = 5
n_neuron_L2 = 10

n_pattern_1 = int(Capacity[7,7]*(n_neuron+n_neuron_L2)/2)
n_pattern_2 = int(Capacity[0,3]*(n_neuron+n_neuron_L2)/2)
n_pattern_3 = int(Capacity[0,7]*(n_neuron+n_neuron_L2)/2)





parameter_sets = [
    {"perc_active_L1": 0.5, "perc_active_L2": 0.5,  "n_pattern": n_pattern_1},
    {"perc_active_L1": 0.1, "perc_active_L2": 0.01, "n_pattern": n_pattern_2},
    {"perc_active_L1": 0.5, "perc_active_L2": 0.01, "n_pattern": n_pattern_3}
]
colorsets = ['y', 'r', 'b']

statistics = []

fig, axes = plt.subplots(3,3, figsize = [19,10])
for index, paras in enumerate(parameter_sets):
    
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

    # if not success:
    #     print("The training is not successful")
    bins = np.arange(0,1.01,0.01)
    utils_num.index_analysis_v4(W_12_trained, W_21_trained.mT, patterns_L1=patterns_L1, patterns_L2= patterns_L2, perc_active_L1=perc_active_L1, \
                                            neuron_base_state = neuron_base_state, ax = axes[0,0], bins=bins, color=colorsets[parameter_sets.index(paras)])
    
    plt.xlim([0.25,1])
    plt.xticks(np.arange(0.25,1.1,0.25))
    
    # visualize W_ij+W_ji:
    plt.sca(axes[index,2])
    W_ij_plus_W_ji = W_12_trained + W_21_trained.mT
    W_ij_plus_W_ji = W_ij_plus_W_ji.detach().numpy()
    plt.imshow(W_ij_plus_W_ji, cmap='PiYG', vmin=-1, vmax=1)
    plt.colorbar()

    plt.sca(axes[parameter_sets.index(paras),1])
    # plot the scatter plot of the weight distribution:
    #plt.scatter(W_12_trained.detach().numpy().flatten(), W_21_trained.mT.detach().numpy().flatten(), alpha=0.5, s=0.5, c=colorsets[parameter_sets.index(paras)])
    # plot the scatter plot as a 2D histogram/heatmap:
    plt.hist2d(W_12_trained.detach().numpy().flatten(), W_21_trained.mT.detach().numpy().flatten(), bins=50, cmap='Blues', density=True)
    plt.colorbar()
    plt.xlabel('W_12')
    plt.ylabel('W_21')
    # plot a line at x = 0
    plt.axvline(x=0, color='k', linestyle='--')
    # plot a line at y = 0
    plt.axhline(y=0, color='k', linestyle='--')
    # statistics:
    corr,p = scipy.stats.pearsonr(W_12_trained.detach().numpy().flatten(), W_21_trained.mT.detach().numpy().flatten())
    plt.title('correlation = {0}, p = {1}, perc_active_L1 = {2}, perc_active_L2 = {3}'.format(corr, p, perc_active_L1, perc_active_L2))
    plt.gca().set_aspect('equal')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])


plt.sca(axes[0,2])
# plot a line at x = 0
plt.axvline(x=0, color='k', linestyle='--')

plt.show(block=False)
utils.print_parameters_on_plot(axes[2,0], pars)

plt.savefig('results/index_analysis_combined_hist'+utils.make_name_with_time()+'.pdf')

print("end of index analysis")



print('end')