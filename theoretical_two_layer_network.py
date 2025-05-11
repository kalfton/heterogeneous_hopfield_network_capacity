from scipy.integrate import quad
from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils_basic as utils
from utils_theoretical import equations_simplified
from utils_theoretical import equations
import pickle
import argparse


def theoretical_two_layer_network_capacity(ratio_active_L1, ratio_active_L2, n_neuron_L1, n_neuron_L2, use_reg=False, kappa=0, make_plot=True):
    """
    Calculate the theoretical capacity of a two-layer network with given activation ratios and number of neurons.
    The function uses the equations defined in utils_theoretical.py to calculate the capacity and information capacity.
    It also generates plots of the capacity and information capacity.
    """
    initial_guess = [1, 0]
    m_L1_values = ratio_active_L1*2-1
    m_L2_values = ratio_active_L2*2-1
    
    connection_ratio_L1 = n_neuron_L2/(n_neuron_L1+n_neuron_L2) # the ratio of number of connections vs. total neurons
    connection_ratio_L2 = n_neuron_L1/(n_neuron_L1+n_neuron_L2) # the ratio of number of connections vs. total neurons

    if not use_reg:
        Capacity = np.zeros((len(m_L2_values), len(m_L1_values)))
        for i in range(len(m_L1_values)):
            for j in range(len(m_L2_values)):
                sol1 = root(equations_simplified, initial_guess, args=(m_L1_values[i]), method='hybr') # solve for the capacity of layer 1
                sol2 = root(equations_simplified, initial_guess, args=(m_L2_values[j]), method='hybr') # solve for the capacity of layer 2
                Capacity[j][i] = np.min([ connection_ratio_L1*sol1.x[0], connection_ratio_L2*sol2.x[0]])

        information_capcity_per_neuron_square = np.zeros((len(m_L2_values), len(m_L1_values)))
        for i in range(len(m_L1_values)):
            for j in range(len(m_L2_values)):
                information_capcity_per_neuron_square[j][i] = Capacity[j][i]*n_neuron_L1/(n_neuron_L1+n_neuron_L2)*utils.information_entropy(ratio_active_L1[i]) # the information capacity per neuron

        with open("results/theoretical_two_layer_capacity_L1_"+ str(n_neuron_L1)+ "_L2_"+ str(n_neuron_L2)+".pkl", "wb") as f:
            pickle.dump([Capacity, information_capcity_per_neuron_square, ratio_active_L1, ratio_active_L2], f)
            
    elif use_reg:
        # thoeoretical capacity with kappa
        Capacity = np.zeros((len(m_L2_values), len(m_L1_values)))
        for i in range(len(m_L1_values)):
            for j in range(len(m_L2_values)):
                sol1 = root(equations, initial_guess, args=(kappa, m_L1_values[i]), method='hybr') # solve for the capacity of layer 1
                sol2 = root(equations, initial_guess, args=(kappa, m_L2_values[j]), method='hybr') # solve for the capacity of layer 2
                Capacity[j][i] = np.min([ connection_ratio_L1*sol1.x[0], connection_ratio_L2*sol2.x[0]])

        information_capcity_per_neuron_square = np.zeros((len(m_L2_values), len(m_L1_values)))
        for i in range(len(m_L1_values)):
            for j in range(len(m_L2_values)):
                information_capcity_per_neuron_square[j][i] = Capacity[j][i]*n_neuron_L1/(n_neuron_L1+n_neuron_L2)*utils.information_entropy(ratio_active_L1[i]) # the information capacity per neuron

        with open("results/theoretical_two_layer_capacity_L1_"+ str(n_neuron_L1)+ "_L2_"+ str(n_neuron_L2)+ "_k_"+ str(kappa) +".pkl", "wb") as f:
            pickle.dump([Capacity, information_capcity_per_neuron_square, ratio_active_L1, ratio_active_L2], f)

    # make the plots:
    if make_plot:
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
        h = plt.contourf(xx, yy, Capacity_plot, cmap='viridis', vmin=vmin, vmax=vmax, levels = np.arange(vmin, vmax+0.001, step), norm= utils.Rescaled_Norm(vmin = vmin, vmax = vmax))
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
        h = plt.contourf(xx, yy, information_capcity_per_neuron_square, cmap='viridis', vmin=vmin, vmax=vmax, levels = np.arange(vmin, vmax+0.001, step), norm= utils.Rescaled_Norm(vmin = vmin, vmax = vmax))
        plt.axis('scaled')
        plt.colorbar()
        plt.xlabel("Activation probability in layer 1")
        plt.ylabel("Activation probability in layer 2")
        # make the plot square:
        plt.gca().set_aspect('auto')
        plt.show(block=False)

        if not use_reg:
            plt.savefig("results/theoretical_two_layer_capacity_L1" + str(n_neuron_L1)+ "_L2_"+ str(n_neuron_L2)+ utils.make_name_with_time()+".pdf")
        elif use_reg:
            plt.savefig("results/theoretical_two_layer_capacity_kappa_" +str(kappa)+ "_ratioL1_"  + str(n_neuron_L1)+ "_ratioL2_"+ str(n_neuron_L2)+ utils.make_name_with_time()+".pdf")

        # Multiply the capacity and information capacity.
        weight_lambda = 1.0
        mixed_capacity = Capacity*(information_capcity_per_neuron_square** weight_lambda)
        # plot the mixed capacity
        mixed_capacity = mixed_capacity/np.max(mixed_capacity)

        vmax=1
        vmin=0.0
        step = 0.01
        plt.figure(figsize=(10, 8))
        Capacity_plot = mixed_capacity.copy()
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
        plt.savefig("results/theoretical_two_layer_joint_capacity_kappa_" +str(kappa)+ "_L1_" + str(n_neuron_L1)+ "_L2_"+ str(n_neuron_L2) + utils.make_name_with_time()+".pdf")

    return Capacity, information_capcity_per_neuron_square, ratio_active_L1, ratio_active_L2


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Two layer network finite size analysis')
    parser.add_argument('--n_neuron', type=int, default=150, help='number of neurons in layer 1')
    parser.add_argument('--n_neuron_L2', type=int, default=450, help='number of neurons in layer 2')
    parser.add_argument('--use_reg', action='store_true', help='whether to use regularization with finite kappa')
    parser.add_argument('--kappa', type=float, default=0.0, help='the kappa for the training')
    args = parser.parse_args()
    pars=vars(args)
    
    use_reg = pars['use_reg']
    # looping through different values of m and m_2 and make a plot of a_c and M_c
    ratio_active_L1 = np.concatenate((np.arange(0.01, 0.0999, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.999, 0.03)))
    ratio_active_L2 = np.concatenate((np.arange(0.01, 0.0999, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.999, 0.03)))
    kappa=pars['kappa']
    n_neuron_L1 = pars['n_neuron']
    n_neuron_L2 =  pars['n_neuron_L2']

    theoretical_two_layer_network_capacity(ratio_active_L1, ratio_active_L2, n_neuron_L1, n_neuron_L2, use_reg, kappa, make_plot=True)