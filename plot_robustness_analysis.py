import numpy as np
import torch
from matplotlib import pyplot as plt
import utils_basic as utils
import utils_numerical as utils_num
import pickle
import warnings

def plot_robustness_analysis(robustness_score, hipp_index_score, information_capacity_theoretical, capacities_theoretical,\
                             robustness_score_ideal, hipp_index_score_ideal, capacities_ideal, information_capacity_ideal, perc_act_L1_set, perc_act_L2_set, pars):
    fig, axes = plt.subplots(3,3, figsize = [19,10])

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.4, hspace=0.4)

    utils_num.heatmap_plot(perc_act_L1_set, perc_act_L2_set, robustness_score, \
                "Activation probability in layer 1", "Activation probabiltiy in layer 2", "Robustness score", "The robustness score of the two-layer network", ax=axes[0,0], vmax = 1.0, vmin = 0.5, cmap='inferno', rescale_norm=False)
    utils_num.heatmap_plot(perc_act_L1_set, perc_act_L2_set, hipp_index_score,\
                "Activation probability in layer 1", "Activation probabiltiy in layer 2", "Hippo index score", "Hippo index score", step = 0.005, ax=axes[0,1])
    # set the color bar ticks:
    # cbar = plt.colorbar(axes[0,1].collections[0], ax=axes[0,1])
    # cbar.set_ticks(np.arange(0.5, 0.61, 0.02))

    # plot the scatter of the lower branch:
    midpoint = int(len(perc_act_L2_set)/2)
    Info_times_memory_theoretical = information_capacity_theoretical*capacities_theoretical
    normalization_factor = np.max(Info_times_memory_theoretical)
    # normalize the joint capacity:
    Info_times_memory_theoretical = Info_times_memory_theoretical/normalization_factor
    Info_times_memory_ideal = information_capacity_ideal*capacities_ideal/normalization_factor

    utils_num.scatter_plot_color(Info_times_memory_theoretical[0:midpoint,:].flatten(), robustness_score[0:midpoint,:].flatten(), hipp_index_score[0:midpoint,:].flatten(), \
                        "Information times memory", "Robustness score", "Hippo index score", "Info times memory theoretical", ax=axes[0,2], square=False)
    # set the color bar ticks:
    # cbar = plt.colorbar(axes[0,2].collections[0], ax=axes[0,2])
    # cbar.set_ticks(np.arange(0.5, 0.61, 0.02))
    plt.sca(axes[0,2])
    plt.scatter(Info_times_memory_ideal, robustness_score_ideal, c=hipp_index_score_ideal, cmap='cool', vmin=0, vmax=1, s=12)
    axes[0,2].set_aspect(2.5)

    utils.print_parameters_on_plot(axes[2,2], pars)

    plt.show(block=False)
    plt.savefig('results/robustness_score_two_layer'+utils_num.make_name_with_time()+'.pdf')


if __name__ == "__main__":
    with open(f'results/robustness_score_two_layer_ablation=1, n_layer1 = 50, n_layer_2 = 150.pkl', 'rb') as f:
        results = pickle.load(f)
    robustness_score = results["robustness_score"]
    hipp_index_score = results["hipp_index_score"]
    capacities_theoretical = results["capacities_theoretical"]
    information_capacity_theoretical = results["information_capacity_theoretical"]
    robustness_score_ideal = results["robustness_score_ideal"]
    hipp_index_score_ideal = results["hipp_index_score_ideal"]
    capacities_ideal = results["capacities_ideal"]
    information_capacity_ideal = results["information_capacity_ideal"]
    perc_act_L1_set = results["perc_act_L1_set"]
    perc_act_L2_set = results["perc_act_L2_set"]
    pars = results["pars"]

    plot_robustness_analysis(robustness_score, hipp_index_score, information_capacity_theoretical, capacities_theoretical,\
                             robustness_score_ideal, hipp_index_score_ideal, capacities_ideal, information_capacity_ideal, perc_act_L1_set, perc_act_L2_set, pars)
