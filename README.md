
# network_capacity

This repository contains the code used in our paper on the memory capacity of heterogeneous Hopfield networks.

## How to Run the Analyses

To run the analyses, execute the following command in a Bash shell terminal:

```bash
./Run_analyses.sh
````

Parameters can be customized directly within the script.

## Code Descriptions

* [`numerical_perceptron_capacity_parallel.py`](./numerical_perceptron_capacity_parallel.py)
  Numerically estimates the memory capacity of perceptrons and compares the result with analytical predictions.

* [`numerical_random_network_cm_parallel.py`](./numerical_random_network_cm_parallel.py)
  Estimates the capacity of randomly connected heterogeneous networks and compares it with analytical results.
  It can also estimate the capacity of randomly connected committee machine networks by changing parameter settings.

* [`numerical_two_layer_network_parallel.py`](./numerical_two_layer_network_parallel.py)
  Estimates the memory capacity of a two-layer network (e.g., the DG–CA3 model).

* [`theoretical_two_layer_network.py`](./theoretical_two_layer_network.py)
  Calculates the analytical capacity of a two-layer network as a function of the coding levels in both layers.

* [`robustness_analysis.py`](./robustness_analysis.py)
  Quantifies the robustness of stored memory patterns under neuron ablation in a two-layer network.

* [`index_score_heatmap.py`](./index_score_heatmap.py) and [`index_score_distribution.py`](./index_score_distribution.py)
  Compute the Network Index Score, inspired by the hippocampal indexing theory.

* [`two_layer_finite_size_analysis.py`](./two_layer_finite_size_analysis.py)
  Demonstrates how the numerical capacity deviates from analytical predictions in finite-size networks.


