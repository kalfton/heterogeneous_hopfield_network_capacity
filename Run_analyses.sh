#!/bin/bash

echo "start the analyses"
N_input=100 # number of input neurons in perceptron
N_neuron=100 # number of neurons in the heterogeneous random networks
N_neuron_small=50 # number of neurons in the analyses which are computationally heavy.

N_neuron_L1=50 # number of neurons in the first layer of the two-layer network
N_neuron_L2=150 # number of neurons in the second layer of the two-layer network

N_repeat=2 # number of independent and repetitive measurements in numerical simulations
N_process=12 # number of cores for parallelization in the calculations to speed up the simulations, increase this number if you have more cores available

# # numerically calulate the capacity of the perceptron and compare it with the analytical solution
python numerical_perceptron_capacity_parallel.py --n_neuron ${N_input} --n_repeat ${N_repeat} --n_process ${N_process} --add_bias

# # numerically calculate the capacity of the heterogeneous random network and compare it with the analytical solution
python numerical_random_network_cm_parallel.py --n_neuron ${N_neuron} --method svm --change_in_degree --n_repeat ${N_repeat} --n_process ${N_process} --pattern_act_mean 0.25 --ratio_conn_mean 0.5
python numerical_random_network_cm_parallel.py --n_neuron ${N_neuron} --method svm --change_out_degree --n_repeat ${N_repeat} --n_process ${N_process} --pattern_act_mean 0.25 --ratio_conn_mean 0.5
python numerical_random_network_cm_parallel.py --n_neuron ${N_neuron} --method svm --change_in_degree --n_repeat ${N_repeat} --n_process ${N_process} --pattern_act_mean 0.25 --ratio_conn_mean 0.5 --use_reg --kappa 0.5

# # analyze in the trained heterogeneous random network the relation of coding levels, the ratio of positive and negative inputs, and the threshold. 
python pos_neg_ratio_in_heter_network.py --n_neuron ${N_neuron} --n_repeat ${N_repeat} --pattern_act_mean 0.25  --ratio_conn_mean 0.5 --change_in_degree --heter_type both_uncorr --ratio_conn_std 0.10 --pattern_act_std 0.10
python pos_neg_ratio_in_heter_network.py --n_neuron ${N_neuron} --n_repeat ${N_repeat} --pattern_act_mean 0.25  --ratio_conn_mean 0.5 --change_in_degree --heter_type both_positive_corr --ratio_conn_std 0.10 --pattern_act_std 0.10
python pos_neg_ratio_in_heter_network.py --n_neuron ${N_neuron} --n_repeat ${N_repeat} --pattern_act_mean 0.25  --ratio_conn_mean 0.5 --change_in_degree --heter_type both_negative_corr --ratio_conn_std 0.10 --pattern_act_std 0.10

# # numerically analyze the capacity of the heterogeneous random network to store examples and concepts
python memorize_example_concept_parallel.py --n_neuron ${N_neuron_small} --n_repeat ${N_repeat} --reinit_prop 0.3 --pattern_act_mean 0.25  --ratio_conn_mean 0.5 --change_in_degree --heter_type both_uncorr --n_process ${N_process}
python memorize_example_concept_parallel.py --n_neuron ${N_neuron_small} --n_repeat ${N_repeat} --reinit_prop 0.3 --pattern_act_mean 0.25  --ratio_conn_mean 0.5 --change_in_degree --heter_type both_positive_corr --n_process ${N_process}
python memorize_example_concept_parallel.py --n_neuron ${N_neuron_small} --n_repeat ${N_repeat} --reinit_prop 0.3 --pattern_act_mean 0.25  --ratio_conn_mean 0.5 --change_in_degree --heter_type both_negative_corr --n_process ${N_process}

# # numerically calculte the capacity of the heterogeneous random network with each unit as a committee machine
python numerical_random_network_cm_parallel.py --n_neuron ${N_neuron_small} --method backprop --change_in_degree --n_repeat ${N_repeat} --n_process ${N_process}  --pattern_act_mean 0.25 --training_max_iter 10000 --n_dentrates 3 --use_committee_machine --lr_committee 0.05 --add_bias


# # analytically calculate the capacity of two layer networks
python theoretical_two_layer_network.py --n_neuron ${N_neuron_L1} --n_neuron_L2 ${N_neuron_L2}
# # numerically calculate the capacity of two layer networks
python numerical_two_layer_network_parallel.py --n_neuron ${N_neuron_L1} --n_neuron_L2 ${N_neuron_L2} --method svm --training_max_iter 1000 --n_repeat ${N_repeat} --n_process ${N_process}


# # calculate and plot the hippocampal index score in the two-layer network
python index_score_heatmap.py --n_neuron ${N_neuron_L1} --n_neuron_L2 ${N_neuron_L2} --method svm --training_max_iter 1000 --n_repeat ${N_repeat}
python index_score_distribution.py --n_neuron ${N_neuron_L1} --n_neuron_L2 ${N_neuron_L2} --method svm


# # calculate the robustness score and plot its relation with the hippocampal index score and the capacities.
python robustness_analysis_parallel.py --n_neuron ${N_neuron_L1} --n_neuron_L2 ${N_neuron_L2}

# # finite size analysis of the two-layer network
python two_layer_finite_size_analysis.py --method svm --training_max_iter 1000 --n_repeat ${N_repeat} --n_process ${N_process}
python two_layer_finite_size_plot.py --n_repeat ${N_repeat}

