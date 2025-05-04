from scipy.integrate import quad
from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils_basic as utils
from utils_theoretical import equations_simplified
from utils_theoretical import equations
import pickle

initial_guess = [1, 1]
use_kappa = False

# looping through different values of m and m_2 and make a plot of a_c and M_c
ratio_active_L1 = np.concatenate((np.arange(0.01, 0.0999, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.999, 0.03)))
ratio_active_L2 = np.concatenate((np.arange(0.01, 0.0999, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.999, 0.03)))
k_value = 0.0
m_L1_values = ratio_active_L1*2-1
m_L2_values = ratio_active_L2*2-1

n_neuron_L1 = 450
n_neuron_L2 =  150
connection_ratio_L1 = n_neuron_L2/(n_neuron_L1+n_neuron_L2) # the ratio of number of connections vs. total neurons
connection_ratio_L2 = n_neuron_L1/(n_neuron_L1+n_neuron_L2) # the ratio of number of connections vs. total neurons
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

plt.savefig("results/theoretical_two_layer_capacity_L1" + str(n_neuron_L1)+ "_L2_"+ str(n_neuron_L2)+ utils.make_name_with_time()+".pdf")
print("end")

# # save the capacity values as pickle file:
with open("results/theoretical_two_layer_capacity_L1_"+ str(n_neuron_L1)+ "_L2_"+ str(n_neuron_L2)+".pkl", "wb") as f:
    pickle.dump([Capacity, information_capcity_per_neuron_square, ratio_active_L1, ratio_active_L2], f)


if use_kappa:
    # thoeoretical capacity with kappa
    kappa = 0.5

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

    # plt.savefig("results/theoretical_rbm_capacity_0_1_rbm.png")
    plt.savefig("results/theoretical_two_layer_capacity_kappa_%d_ratioL1_" % (kappa) + str(n_neuron_L1)+ "_ratioL2_"+ str(n_neuron_L2)+ utils.make_name_with_time()+".pdf")
    print("end")

    with open("results/theoretical_two_layer_capacity_L1_"+ str(n_neuron_L1)+ "_L2_"+ str(n_neuron_L2)+ "_k_"+ str(kappa) +".pkl", "wb") as f:
        pickle.dump([Capacity, information_capcity_per_neuron_square, ratio_active_L1, ratio_active_L2], f)



# Add the capacity and information capacity up in the plot and find the maximum value of the sum
weight_lambda = 1.0
mixed_capacity = Capacity*(information_capcity_per_neuron_square** weight_lambda)
# find the indices of the maximum value of the sum
max_index = np.unravel_index(np.argmax(mixed_capacity, axis=None), mixed_capacity.shape)

ratio_active_L1_star = ratio_active_L1[max_index[1]]
ratio_active_L2_star = ratio_active_L2[max_index[0]]

max_value = np.max(mixed_capacity)
print("weight_lambda: ", weight_lambda) 
print("The ratio to get the maximum value of the sum of capacity and information capacity is: L1: ", ratio_active_L1_star, " , L2:", ratio_active_L2_star)


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
plt.savefig("results/theoretical_two_layer_mixed_capacity_multiplication_L1_" + str(n_neuron_L1)+ "_L2_"+ str(n_neuron_L2) + utils.make_name_with_time()+".pdf")
print("end")
