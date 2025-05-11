import numpy as np
import torch
from matplotlib import pyplot as plt
import utils_basic as utils
from matplotlib.patches import Patch
import pickle
import warnings
import time
import argparse
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot the finite size effect of two layer network')
    parser.add_argument('--W_symmetric', action='store_true', help='whether the W matrix is symmetric')
    parser.add_argument('--n_repeat', type=int, default=50, help='number of repeats')
    parser.add_argument('--kappa', type=float, default=0, help='kappa value')

    args = parser.parse_args()
    pars = vars(args)

    perc_act_L1_set = np.arange(0.01, 0.52, 0.02)
    perc_act_L2_set = np.arange(0.01, 0.21, 0.01)
    kappa = pars["kappa"]
    n_repeat = pars["n_repeat"]

    n_neuron_L1_set = np.array([10, 15, 25, 50, 75])
    n_neuron_L2_set = 3*n_neuron_L1_set

    # load the results:
    arg_max_perc_act_L1 = np.zeros(len(n_neuron_L1_set))
    arg_max_perc_act_L2 = np.zeros(len(n_neuron_L1_set))
    max_capacities = np.zeros(len(n_neuron_L1_set))
    capacities_matrix = np.zeros((len(n_neuron_L1_set), len(perc_act_L2_set), len(perc_act_L1_set)))


    arg_max_perc_act_L1_info = np.zeros(len(n_neuron_L1_set))
    arg_max_perc_act_L2_info = np.zeros(len(n_neuron_L1_set))
    max_capacities_info = np.zeros(len(n_neuron_L1_set))
    information_capacity_matrix = np.zeros((len(n_neuron_L1_set), len(perc_act_L2_set), len(perc_act_L1_set)))


    arg_max_perc_act_L1_mixed = np.zeros(len(n_neuron_L1_set))
    arg_max_perc_act_L2_mixed = np.zeros(len(n_neuron_L1_set))
    max_capacities_mixed = np.zeros(len(n_neuron_L1_set))
    capacities_matrix_mixed = np.zeros((len(n_neuron_L1_set), len(perc_act_L2_set), len(perc_act_L1_set)))
    for iii in range(len(n_neuron_L1_set)):
        with open('results/numerical_two_layer_finite_size_'+'k = %.1f, n_layer1 = %d, n_layer_2 = %d, repeat = %d' % (kappa, n_neuron_L1_set[iii], n_neuron_L2_set[iii], n_repeat) +'.pkl', 'rb') as f:
            capacities, information_capacity, _, _ ,pars = pickle.load(f)
            # Only get the values with perc_act_L1<=0.51

            capacities = capacities[:,:len(perc_act_L1_set)]
            information_capacity = information_capacity[:,:len(perc_act_L1_set)]
            capacities_matrix[iii,:,:] = capacities
            i_max, j_max = np.unravel_index(np.argmax(capacities, axis=None), capacities.shape)
            arg_max_perc_act_L1[iii] = perc_act_L1_set[j_max]
            arg_max_perc_act_L2[iii] = perc_act_L2_set[i_max]
            max_capacities[iii] = capacities[i_max, j_max]


            information_capacity_matrix[iii,:,:] = information_capacity
            i_max_info, j_max_info = np.unravel_index(np.argmax(information_capacity, axis=None), information_capacity.shape)
            arg_max_perc_act_L1_info[iii] = perc_act_L1_set[j_max_info]
            arg_max_perc_act_L2_info[iii] = perc_act_L2_set[i_max_info]
            max_capacities_info[iii] = information_capacity[i_max_info, j_max_info]

            capacities_mixed = capacities*information_capacity
            capacities_matrix_mixed[iii,:,:] = capacities_mixed
            i_max_mixed, j_max_mixed = np.unravel_index(np.argmax(capacities_mixed, axis=None), capacities_mixed.shape)
            arg_max_perc_act_L1_mixed[iii] = perc_act_L1_set[j_max_mixed]
            arg_max_perc_act_L2_mixed[iii] = perc_act_L2_set[i_max_mixed]
            max_capacities_mixed[iii] = capacities_mixed[i_max_mixed, j_max_mixed]


    # 2D countour plot for finite size effect:

    percent_threshold = 0.90
    fig, axes = plt.subplots(2,2, figsize = [14,10])
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.4, hspace=0.4)


    colors = plt.cm.cool(np.linspace(0, 1, len(n_neuron_L1_set)))
    colors[:, -1] = 0.3 # set the alpha channel to 0.3
    dot_size = 40
    plt.sca(axes[0,0])
    for i in range(len(n_neuron_L1_set)):
        # binarize capacity matrix:
        threshold = max_capacities[i]*percent_threshold
        capacity_matrix_binary = capacities_matrix[i,:,:] >= threshold
        xx, yy = np.meshgrid(perc_act_L1_set, perc_act_L2_set)
        plt.contourf(xx, yy, capacity_matrix_binary, levels=[-0.5, 0.5, 1.5],colors=[[1,1,1,0.01], colors[i]])

    plt.xticks(np.arange(0.1, 0.51, 0.1))
    plt.xlabel("Activation probability in layer 1")
    plt.ylabel("Activation probability in layer 2")
    plt.title("Finite size effect: Capacity binary contour plot")

    # label the color:
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor='none', label=n_neuron_L1_set[i]*4, alpha=0.5)
        for i in range(len(n_neuron_L1_set))
    ]

    # Add legend to the axis
    axes[0,0].legend(handles=legend_handles, loc='upper right')


    plt.sca(axes[1,0])
    # information capacity plot:
    for i in range(len(n_neuron_L1_set)):
        threshold = max_capacities_info[i]*percent_threshold
        information_capacity_matrix_binary = information_capacity_matrix[i,:,:] >= threshold
        xx, yy = np.meshgrid(perc_act_L1_set, perc_act_L2_set)
        plt.contourf(xx, yy, information_capacity_matrix_binary, levels=[-0.5, 0.5, 1.5],colors=[[1,1,1,0.01], colors[i]])

    plt.xticks(np.arange(0.1, 0.51, 0.1))
    plt.xlabel("Activation probability in layer 1")
    plt.ylabel("Activation probability in layer 2")
    plt.title("Finite size effect: Information capacity binary contour plot")

    plt.sca(axes[0,1])
    # information times capacity plot:
    for i in range(len(n_neuron_L1_set)):
        threshold = max_capacities_mixed[i]*percent_threshold
        capacities_mixed_binary = capacities_matrix_mixed[i,:,:] >= threshold
        xx, yy = np.meshgrid(perc_act_L1_set, perc_act_L2_set)
        plt.contourf(xx, yy, capacities_mixed_binary, levels=[-0.5, 0.5, 1.5],colors=[[1,1,1,0.01], colors[i]])
    plt.xticks(np.arange(0.1, 0.51, 0.1))
    plt.xlabel("Activation probability in layer 1")
    plt.ylabel("Activation probability in layer 2")
    plt.title("Finite size effect: Information times capacity binary contour plot")
        

    # plot the maximum capacity point:
    plt.sca(axes[0,0])
    for i in range(len(n_neuron_L1_set)):
        plt.scatter(arg_max_perc_act_L1[i], arg_max_perc_act_L2[i], s=dot_size, color=colors[i])

    plt.sca(axes[1,0])
    for i in range(len(n_neuron_L1_set)):
        plt.scatter(arg_max_perc_act_L1_info[i], arg_max_perc_act_L2_info[i], s=dot_size, color=colors[i])

    plt.sca(axes[0,1])
    for i in range(len(n_neuron_L1_set)):
        plt.scatter(arg_max_perc_act_L1_mixed[i], arg_max_perc_act_L2_mixed[i], s=dot_size, color=colors[i])


    utils.print_parameters_on_plot(axes[1,1], pars)
    # plt.show(block=False)

    plt.savefig('results/numerical_two_layer_max_capacity_with_finite_size_effect_'+utils.make_name_with_time()+'.pdf')



