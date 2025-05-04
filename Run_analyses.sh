#!/bin/bash

printf "start running"

python numerical_perceptron_capacity_parallel.py --n_neuron 1000 --add_bias --n_process 10 --n_repeat 10

python numerical_random_network_parallel.py --W_notsymmetric --n_neuron 500 --method svm --change_in_degree --n_repeat 10 --n_process 10 --pattern_act_mean 0.25

python numerical_two_layer_network_parallel.py --W_notsymmetric --n_neuron 150 --n_neuron_L2 450 --network_type RBM --method svm --training_max_iter 1000 --neuron_base_state -1 --epsilon 0.01 --n_repeat 10

python theoretical_two_layer_network.py

python index_theory_analysis_V2.py --W_notsymmetric --n_neuron 150 --n_neuron_L2 450 --n_pattern 500 --perc_active_L1 0.5 --perc_active_L2 0.01 --network_type RBM --method svm --neuron_base_state -1 --kappa 0

python index_theory_heatmap.py