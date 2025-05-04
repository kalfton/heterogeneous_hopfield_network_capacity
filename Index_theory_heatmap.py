import numpy as np
from numpy import random
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy
# from two_layer_hopfield_simplified import Two_layer_hopfield
from random_connection_network import random_connect_network
import utils_basic as utils
import utils_numerical as utils_num
import pickle
import warnings
import time
import argparse
import sys

use_kappa = True
n_neuron=150
n_neuron_L2=450

if use_kappa and n_neuron == 150 and n_neuron_L2 == 450:
    with open('results/numerical_two_layer_temp_k = 0.500000, _n_layer1 = 150, n_layer_2 = 450_20250411-214253.pkl', 'rb') as f:
        _, _, perc_act_L1_set, perc_act_L2_set, pars = pickle.load(f)
    with open('results/theoretical_two_layer_capacity_L1_150_L2_450_k_0.5.pkl', 'rb') as f:
        capacities, information_capacity, _, _ = pickle.load(f)
elif n_neuron == 150 and n_neuron_L2 == 450:
    with open('results/numerical_two_layer_temp_k = 0, n_layer1 = 150, n_layer_2 = 450_cluster.pkl', 'rb') as f:
        _, _, perc_act_L1_set, perc_act_L2_set, pars = pickle.load(f)
    with open('results/theoretical_two_layer_capacity_L1_150_L2_450.pkl', 'rb') as f:
        capacities, information_capacity, _, _ = pickle.load(f)
elif n_neuron == 450 and n_neuron_L2 == 150:
    with open('results/numerical_two_layer_temp_k = 0, n_layer1 = 450, n_layer_2 = 150_cluster.pkl', 'rb') as f:
        _, _, perc_act_L1_set, perc_act_L2_set, pars = pickle.load(f)
    with open('results/theoretical_two_layer_capacity_L1_450_L2_150.pkl', 'rb') as f:
        capacities, information_capacity, _, _ = pickle.load(f)

else:
    raise ValueError("The file does not exist")

# # load the parameters for numerical simulation:
# with open('results/numerical_two_layer_temp_k = 0.500000, _n_layer1 = 150, n_layer_2 = 450_20250411-214253.pkl', 'rb') as f:
#     _, _, perc_act_L1_set, perc_act_L2_set, pars = pickle.load(f)

# # load the capacity values from theoretical analysis:
# with open('results/theoretical_two_layer_capacity_L1_150_L2_450_k_0.5.pkl', 'rb') as f:
#     capacities, information_capacity, _, _ = pickle.load(f)

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
            if pars["use_kappa"]:
                success, stored_patterns_all = network.train_svm(patterns_all, use_reg = True, kappa = pars["kappa"])
            else:
                success, stored_patterns_all = network.train_svm(patterns_all, use_reg = False)


        # index theory analysis:
        W_12_trained = network.weight.clone().numpy()[:n_neuron, n_neuron:]
        W_21_trained = network.weight.clone().numpy()[n_neuron:, :n_neuron]

        dists = utils_num.index_analysis_v4(W_12_trained, W_21_trained.T, patterns_L1=patterns_L1,patterns_L2=patterns_L2, perc_active_L1=perc_act_L1_set[j], neuron_base_state = neuron_base_state, makeplot=False)
        hipp_index_score_1[i,j] = dists
            

        # # for debug:
        if np.isnan(dists) or dists < 0:
            print("Error in hipp_index_score_1[%d,%d]" % (i,j))

with open('results/numerical_two_layer_hippo_index_L1'+str(n_neuron)+'_L2'+str(n_neuron_L2)+'_kappa'+str(pars["kappa"])+'.pkl', 'wb') as f:
    pickle.dump([hipp_index_score_1, pars], f)

# with open('results/numerical_two_layer_hippo_index_L1'+str(n_neuron)+'_L2'+str(n_neuron_L2)+'_kappa'+str(pars["kappa"])+'.pkl', 'rb') as f:
#     hipp_index_score_1, pars = pickle.load(f)

# plot the index score:
fig, axes = plt.subplots(2,2, figsize = [19,10])

plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.4, hspace=0.4)


utils_num.heatmap_plot(perc_act_L1_set, perc_act_L2_set, hipp_index_score_1, x_label="Activation probability in layer 1", y_label="Activation probability in layer 2",\
                       z_label="Network index score", title = "Index score", step = 0.005, ax=axes[0,0])

# plt.sca(axes[0,0])
# vmax = np.ceil(np.nanmax(hipp_index_score_1)*100)/100
# vmin = np.floor(np.nanmin(hipp_index_score_1)*100)/100 #0.0 
# step = 0.01
# index_for_plot = hipp_index_score_1
# # index_for_plot[index_for_plot<vmin] = vmin
# # index_for_plot[index_for_plot>vmax] = vmax
# xx, yy = np.meshgrid(perc_act_L1_set, perc_act_L2_set)
# h = plt.contourf(xx, yy, index_for_plot, cmap='viridis', vmin=vmin, vmax=vmax, levels = np.arange(vmin, vmax+0.001, step), norm=utils_num.Rescaled_Norm(vmax=vmax, vmin = vmin))
# # label the colorbar:
# cbar = plt.colorbar()
# cbar.set_label("hippo index score 1", rotation=270, labelpad=20)
# cbar.set_ticks(np.arange(vmin, vmax+0.001, 0.1))
# plt.xlabel("Activation probability in layer 1")
# plt.ylabel("Activation probabiltiy in layer 2")
# plt.title("The distance between the similarity of closest pattern and the similarity of the other patterns")
# # make the axis square:
# plt.gca().set_aspect('equal', adjustable='box')


utils.print_parameters_on_plot(axes[1,1], pars)
plt.show(block=False)
plt.savefig('results/numerical_two_layer_hippo_index' +utils.make_name_with_time()+'.pdf')

print("end")
