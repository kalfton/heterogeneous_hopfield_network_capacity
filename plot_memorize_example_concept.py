import numpy as np
import torch
from matplotlib import pyplot as plt
import utils_basic as utils
import pickle
import warnings
import matplotlib
from matplotlib.patches import Patch

def plot_example_correlated_concepts(results):
    """ plot the 3D surface of example error = epsilon and concept error = epsilon in the 3D space of n_concept, n_example_per_concept, and network_heterogeneity"""

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
    n_neuron = pars['n_neuron']
    epsilon = 0.01

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

if __name__ == "__main__":
    # Load the results from the pickle file
    with open('results/training_error_concept_and_example_3D_matrix_20250330-201739.pkl', 'rb') as f:
        results = pickle.load(f)

    # Call the function to plot the results
    plot_example_correlated_concepts(results)