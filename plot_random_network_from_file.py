import numpy as np
from numpy import random
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import scipy
from utils_theoretical import theoretical_network_capacity
from utils_theoretical import theoretical_network_capacity_kappa
import utils_basic as utils
import pickle
import warnings
import time
import argparse
import sys



# load the results:
with open('results/Heterogeneous_committee_machine_network_capacity_k=0_input_heter_=1_20250315-112747.pkl', 'rb') as f:
    capacities, capacities_theoretical, capacities_matched, capacities_theoretical_matched, capacities_anti_matched, capacities_theoretical_anti_matched, ratio_conn_std_list, pattern_act_std_list, pars = pickle.load(f)
# make the colorbar scale the same for all subplots:
vmin=0.0
vmax = np.ceil(np.max([np.max(capacities), np.max(capacities_matched), np.max(capacities_anti_matched)])*10)/10+0.01
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