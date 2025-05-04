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
from matplotlib.patches import Patch
matplotlib.use('TkAgg')



# plot the 3D surface of example error = epsilon and concept error = epsilon in the 3D space of n_concept, n_example_per_concept, and network_heterogeneity
with open('results/training_error_concept_and_example_3D_matrix_20250330-201739.pkl', 'rb') as f:
    results = pickle.load(f)

n_concept_set = results['n_concept_set']
n_example_per_concept_set = results['n_example_per_concept_set']
network_heter_level_set = results['network_heter_level_set']
coding_level_heter_set = results['coding_level_heter_set']
example_errors = results['example_errors']
concept_errors = results['concept_errors']
pars = results['pars']

if pars["heter_type"] == "coding_level":
    heter_level_set = coding_level_heter_set
else:
    heter_level_set = network_heter_level_set

epsilon = 0.01
fig = plt.figure(figsize=(19, 10))

# 3D subplots
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
# 2D subplot
ax4 = fig.add_subplot(2, 2, 4)

if example_errors.min() < epsilon:
    utils.plot_error_surface_with_coords(example_errors, n_concept_set, n_example_per_concept_set, heter_level_set, level = epsilon, surface_color='red', ax = ax1)
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    ax1.set_title("Example error = %f" % epsilon)
if concept_errors.min() < epsilon:
    utils.plot_error_surface_with_coords(concept_errors, n_concept_set, n_example_per_concept_set, heter_level_set, level = epsilon, surface_color='blue', ax = ax2)
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    ax2.set_title("Concept error = %f" % epsilon)
# plot the two surfaces in the same plot
if example_errors.min() < epsilon and concept_errors.min() < epsilon:
    utils.plot_error_surface_with_coords(example_errors, n_concept_set, n_example_per_concept_set, heter_level_set, level = epsilon, surface_color='red', ax = ax3)
    utils.plot_error_surface_with_coords(concept_errors, n_concept_set, n_example_per_concept_set, heter_level_set, level = epsilon, surface_color='blue', ax = ax3)
    ax3.set_title("Example error = %f and concept error = %f" % (epsilon, epsilon))
# ax3.set_xscale('log')
# ax3.set_yscale('log')
utils.print_parameters_on_plot(ax4, pars)

# save the figure
plt.show(block=False)
# plt.savefig('results/3D_surface_training_error_concept_vs_example'+utils.make_name_with_time()+'.png')

print("done")

# sanity check analysis:
n_neuron = pars['n_neuron']
# calculate the numerical capacity of the network and plot it as a function of network heterogeneity
fig, axes = plt.subplots(3,3, figsize = [19,10])

# capacity = np.argwhere(example_errors[:, 0, :]<epsilon)
capacity = np.zeros((len(heter_level_set)))
n_concept_index = 0
for i in range(len(heter_level_set)):
    capacity_inds = np.argwhere(example_errors[n_concept_index, :, i]<epsilon)
    if len(capacity_inds) == 0:
        capacity[i] = 0
    else:
        capacity[i] = n_example_per_concept_set[capacity_inds[-1][0]] * n_concept_set[n_concept_index]/n_neuron
    

plt.sca(axes[0,0])
plt.plot(heter_level_set, capacity)
plt.xlabel("Network heterogeneity")
plt.ylabel("Numerical capacity")

# plot two slides of the error matrix as heatmaps
vmin=0
vmax=0.5
step = 0.01

plt.sca(axes[0,1])
xx, yy = np.meshgrid(n_concept_set/n_neuron, n_example_per_concept_set)
h = plt.contourf(xx, yy, example_errors[:,:,0].T, cmap='viridis', vmin=vmin, vmax=vmax,levels = np.arange(vmin, vmax, step))
cbar = plt.colorbar()
cbar.set_label("Example error", rotation=270, labelpad=20)
plt.xlabel("Number of concepts per neuron")
plt.ylabel("Number of examples per concept")
plt.xscale('log')
plt.yscale('log')
plt.title(f'Heterogeneity = {heter_level_set[0]}')

plt.sca(axes[1,1])
xx, yy = np.meshgrid(n_concept_set/n_neuron, n_example_per_concept_set)
h = plt.contourf(xx, yy, example_errors[:,:,-1].T, cmap='viridis', vmin=vmin, vmax=vmax,levels = np.arange(vmin, vmax, step))
cbar = plt.colorbar()
cbar.set_label("Example error", rotation=270, labelpad=20)
plt.xlabel("Number of concepts per neuron")
plt.ylabel("Number of examples per concept")
plt.xscale('log')
plt.yscale('log')
plt.title(f"Heterogeneity = {heter_level_set[-1]}")


plt.sca(axes[0,2])
xx, yy = np.meshgrid(n_concept_set/n_neuron, n_example_per_concept_set)
h = plt.contourf(xx, yy, concept_errors[:,:,0].T, cmap='viridis', vmin=vmin, vmax=vmax,levels = np.arange(vmin, vmax, step))
cbar = plt.colorbar()
cbar.set_label("Concept error", rotation=270, labelpad=20)
plt.xlabel("Number of concepts per neuron")
plt.ylabel("Number of examples per concept")
plt.xscale('log')
plt.yscale('log')
plt.title(f'Heterogeneity = {heter_level_set[0]}')

plt.sca(axes[1,2])
xx, yy = np.meshgrid(n_concept_set/n_neuron, n_example_per_concept_set)
h = plt.contourf(xx, yy, concept_errors[:,:,-1].T, cmap='viridis', vmin=vmin, vmax=vmax,levels = np.arange(vmin, vmax, step))
cbar = plt.colorbar()
cbar.set_label("Concept error", rotation=270, labelpad=20)
plt.xlabel("Number of concepts per neuron")
plt.ylabel("Number of examples per concept")
plt.xscale('log')
plt.yscale('log')
plt.title(f"Heterogeneity = {heter_level_set[-1]}")


utils.print_parameters_on_plot(axes[1,0], pars)



plt.show(block=False)
plt.savefig('results/2D_training_error_concept_vs_example_low_heter_vs_high'+utils.make_name_with_time()+'.pdf')
print("done")


# set a threshold of retrieval and plot three curves representing the capacity of the network in 3 heterogeneity levels

fig, axes = plt.subplots(3,3, figsize = [14,10])
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.4, hspace=0.4)

heter_level_inds = [0, int(len(heter_level_set)/2), -1]
colors_example = plt.cm.spring(np.linspace(0.3, 1, len(heter_level_inds)))
colors_example[:, -1] = 0.3
colors_concept = plt.cm.winter(np.linspace(0.3, 1, len(heter_level_inds)))
colors_concept[:, -1] = 0.3

plt.sca(axes[0,0])

for i, heter_index in enumerate(heter_level_inds):
    # binarize the example error matrix:
    example_errors_binary = example_errors[:,:,heter_index]<epsilon
    xx, yy = np.meshgrid(n_concept_set/n_neuron, n_example_per_concept_set)
    plt.contourf(xx, yy, example_errors_binary.T, levels=[-0.5, 0.5, 1.5],colors=[[1,1,1,0.01], colors_example[i]])
    plt.contour(xx, yy, example_errors_binary.T, levels=[-0.5, 0.5, 1.5],colors=[colors_example[i]],  alpha=0.5, linewidths=2, linestyles = 'dashed')

    concept_errors_binary = concept_errors[:,:,heter_index]<epsilon
    xx, yy = np.meshgrid(n_concept_set/n_neuron, n_example_per_concept_set)
    plt.contourf(xx, yy, concept_errors_binary.T, levels=[-0.5, 0.5, 1.5],colors=[[1,1,1,0.01], colors_concept[i]])
    plt.contour(xx, yy, concept_errors_binary.T, levels=[-0.5, 0.5, 1.5],colors=[colors_concept[i]], alpha=0.5, linewidths=2, linestyles = 'dashed')

# plt.xlabel("Number of concepts per neuron")
# plt.ylabel("Number of examples per concept")
# plt.xscale('log')
# plt.yscale('log')
# plt.title("Example error")

# for i, heter_index in enumerate(heter_level_inds):
#     # binarize the concept error matrix:
#     concept_errors_binary = concept_errors[:,:,heter_index]<epsilon
#     xx, yy = np.meshgrid(n_concept_set/n_neuron, n_example_per_concept_set)
#     plt.contourf(xx, yy, concept_errors_binary.T, levels=[-0.5, 0.5, 1.5],colors=[[1,1,1,0.1], colors_concept[i]], alpha=0.5, zorder=i+4)
#     plt.contour(xx, yy, concept_errors_binary.T, levels=[-0.5, 0.5, 1.5],colors=[colors_concept[i]], alpha=0.5, linewidths=2, linestyles = 'dashed', zorder=i+10)

plt.xlabel("Number of concepts per neuron")
plt.ylabel("Number of examples per concept")
plt.xscale('log')
plt.yscale('log')
plt.title("Retrieval region of Example and Concept")

# label the color:
legend_handles = [
    Patch(facecolor=colors_example[i], edgecolor='none', label=heter_level_set[heter_level_inds[i]], alpha=0.5)
    for i in range(len(heter_level_inds))
]
legend_handles += [Patch(facecolor=colors_concept[i], edgecolor='none', label=heter_level_set[heter_level_inds[i]], alpha=0.5) 
            for i in range(len(heter_level_inds))]

# Add legend to the axis
axes[0,1].legend(handles=legend_handles, loc='upper right')

utils.print_parameters_on_plot(axes[2,2], pars)


plt.show(block=False)
plt.savefig('results/2D_transition_plot_training_error_concept_vs_example'+utils.make_name_with_time()+'.pdf')

print("done")









# # plot the error
# vmin=0
# vmax=0.5
# step = 0.01
# matplotlib.use('TkAgg')
# fig, axes = plt.subplots(2,2, figsize = [19,10])
# # heatmap of example error:
# plt.sca(axes[0,0])
# # plot the meshgrid of capacity matrix:
# xx, yy = np.meshgrid(n_concept_set/n_neuron, n_example_per_concept_set)
# h = plt.contourf(xx, yy, example_errors.T, cmap='viridis', vmin=vmin, vmax=vmax,levels = np.arange(vmin, vmax, step))
# # make the axis to be log scale:
# plt.xscale('log')
# plt.yscale('log')
# # label the colorbar:
# cbar = plt.colorbar()
# cbar.set_label("Example error", rotation=270, labelpad=20)
# plt.xlabel("Number of concepts per neuron")
# plt.ylabel("Number of examples per concept")
# plt.axis('auto')

# # heatmap of concept error:
# plt.sca(axes[0,1])
# # plot the meshgrid of capacity matrix:
# xx, yy = np.meshgrid(n_concept_set/n_neuron, n_example_per_concept_set)
# h = plt.contourf(xx, yy, concept_errors.T, cmap='viridis', vmin=vmin, vmax=vmax,levels = np.arange(vmin, vmax, step))
# # make the axis to be log scale:
# plt.xscale('log')
# plt.yscale('log')
# # label the colorbar:
# cbar = plt.colorbar()
# cbar.set_label("Concept error", rotation=270, labelpad=20)
# plt.xlabel("Number of concepts per neuron")
# plt.ylabel("Number of examples per concept")
# plt.axis('auto')


# utils.print_parameters_on_plot(axes[1,1], pars)

# plt.show(block=False)

# plt.savefig('results/heatmap_training_error_concept_vs_example'+utils.make_name_with_time()+'.png')

# print("done")







