import numpy as np
import matplotlib.pyplot as plt
import utils_basic as utils
import pickle

def plot_two_layer_capacity(Capacity, information_capcity_per_neuron_square, ratio_active_L1, ratio_active_L2, pars):
    """plot numerical capacity, information capacity and joint capacity"""
    weight_lambda = 1.0
    n_neuron_L1 = pars["n_neuron"]
    n_neuron_L2 = pars["n_neuron_L2"]

    Capacity = Capacity/(pars["n_neuron"]+pars["n_neuron_L2"])
    information_capcity_per_neuron_square = information_capcity_per_neuron_square/(pars["n_neuron"]+pars["n_neuron_L2"])**2

    vmax=5.5
    vmin=0.0
    step = 0.1
    plt.figure(figsize=(10, 8))
    Capacity_plot = Capacity.copy()
    Capacity_plot[Capacity_plot>vmax] = vmax-1e-5
    Capacity_plot[Capacity_plot<0] = 0
    plt.subplot(2,2,1)

    # plot the meshgrid of capacity matrix:
    xx, yy = np.meshgrid(ratio_active_L1, ratio_active_L2)
    h = plt.contourf(xx, yy, Capacity_plot, cmap='viridis', vmin=vmin, vmax=vmax, levels = np.arange(vmin, vmax+0.001, step), norm=utils.Rescaled_Norm(vmin = vmin, vmax = vmax))
    plt.axis('scaled')

    cbar = plt.colorbar()
    cbar.set_label("Capacity alpha_N", rotation=270, labelpad=20)

    plt.xlabel("Activation probability in layer 1")
    plt.ylabel("Activation probability in layer 2")
    # make the plot square:
    plt.gca().set_aspect('auto')

    # plot the information capacity
    vmax=0.5
    vmin=0.0
    step = 0.02
    plt.subplot(2,2,2)
    # plot the meshgrid of capacity matrix:
    xx, yy = np.meshgrid(ratio_active_L1, ratio_active_L2)
    h = plt.contourf(xx, yy, information_capcity_per_neuron_square, cmap='viridis', vmin=vmin, vmax=vmax, levels = np.arange(vmin, vmax+0.001, step), norm=utils.Rescaled_Norm(vmin = vmin, vmax = vmax))
    plt.axis('scaled')
    plt.colorbar()
    plt.xlabel("Activation probability in layer 1")
    plt.ylabel("Activation probability in layer 2")
    # make the plot square:
    plt.gca().set_aspect('auto')
    plt.show(block=False)

    plt.savefig("results/numerical_two_layer_capacity_fromfile_ratioL1_" + str(n_neuron_L1)+ "_ratioL2_"+ str(n_neuron_L2)+ utils.make_name_with_time()+".pdf")
    print("end")

    # Joint capacity
    joint_capacity = Capacity*(information_capcity_per_neuron_square** weight_lambda)
    joint_capacity = joint_capacity/np.max(joint_capacity)

    vmax=1
    vmin=0.0
    step = 0.01
    plt.figure(figsize=(10, 8))
    Capacity_plot = joint_capacity.copy()
    Capacity_plot[Capacity_plot>=vmax] = vmax-1e-5
    Capacity_plot[Capacity_plot<0] = 0
    plt.subplot(2,2,1)

    # plot the meshgrid of capacity matrix:
    xx, yy = np.meshgrid(ratio_active_L1, ratio_active_L2)
    h = plt.contourf(xx, yy, Capacity_plot, cmap='viridis', vmin=vmin, vmax=vmax, levels = np.arange(vmin, vmax+0.001, step), norm= utils.Rescaled_Norm(vmin = vmin, vmax = vmax))
    plt.axis('scaled')
    plt.xlabel("Activation probability in layer 1")
    plt.ylabel("Activation probability in layer 2")

    cbar = plt.colorbar()
    cbar.set_label("normalized mix capacity", rotation=270, labelpad=20)
    cbar.set_ticks([0, 0.5, 1.0])
    plt.show(block=False)
    plt.savefig("results/numerical_two_layer_joint_capacity_multiplication_ratioL1_" + str(n_neuron_L1)+ "_ratioL2_"+ str(n_neuron_L2)+ utils.make_name_with_time()+".pdf")
    print("end")


if __name__ == "__main__":
    with open("results/numerical_two_layer_temp_k = 0, _n_layer1 = 150, n_layer_2 = 450_20240625-132928.pkl", "rb") as f:
        [Capacity, information_capcity_per_neuron_square, ratio_active_L1, ratio_active_L2, pars] = pickle.load(f)
    plot_two_layer_capacity(Capacity, information_capcity_per_neuron_square, ratio_active_L1, ratio_active_L2, pars)