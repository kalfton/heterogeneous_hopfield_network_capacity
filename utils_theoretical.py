from scipy.integrate import quad
from scipy.optimize import root
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def integrand1(y, k, M):
    return 1/np.sqrt(2*np.pi) *np.exp(-y**2/2) * (k- M + y)**2

def integrand2(y, k, M):
    return 1/np.sqrt(2*np.pi) *np.exp(-y**2/2) * (k + M + y)**2

def integrand3(y, k, M):
    return 1/np.sqrt(2*np.pi) *np.exp(-y**2/2) * (k - M + y)

def integrand4(y, k, M):
    return 1/np.sqrt(2*np.pi) *np.exp(-y**2/2) * (k + M + y)


def equations(x, k, m_out):
    a, M = x[0], x[1]
    integral1, _ = quad(integrand1, (M-k), np.inf, args=(k,M))
    integral2, _ = quad(integrand2, (-M-k), np.inf, args=(k,M))
    integral3, _ = quad(integrand3, (M-k), np.inf, args=(k,M))
    integral4, _ = quad(integrand4, (-M-k), np.inf, args=(k,M))
    
    eq1 = 1 - a * (((1+m_out)/2) * integral1 + ((1-m_out)/2) * integral2)
    eq2 = ((1+m_out)/2) * integral3 - ((1-m_out)/2) * integral4
    
    return [eq1, eq2]

def integrand5(y, M):
    return 1/np.sqrt(2*np.pi) *np.exp(-y**2/2) * (-M + y)

def integrand6(y, M):
    return 1/np.sqrt(2*np.pi) *np.exp(-y**2/2) * (M + y)

def integrand7(y, M):
    return 1/np.sqrt(2*np.pi) *np.exp(-y**2/2)



# Step 2 & 3: Define the equations with the numerical integrals and solve them
def equations_simplified(x, m_out):
    a, M = x[0], x[1]
    integral5, _ = quad(integrand5, M, np.inf, args=(M))
    integral6, _ = quad(integrand6, -M, np.inf, args=(M))
    integral7, _ = quad(integrand7, -M, M, args=(M))
    
    eq1 = 2 - a * (1-m_out*integral7)
    eq2 = ((1+m_out)/2) * integral5 - ((1-m_out)/2) * integral6
    
    return [eq1, eq2]

def theoretical_perceptron_capacity(m_out_set, kappa):
    initial_guess = [1, 1]
    # this implementation's running time is probably too long, optimize it later.
    perceptron_alpha = np.zeros(len(m_out_set))
    for i in range(len(m_out_set)):
        sol = root(equations, initial_guess, args=(kappa, m_out_set[i]), method='hybr')
        perceptron_alpha[i] = sol.x[0]

    return perceptron_alpha

def theoretical_network_capacity_kappa(weight, neuron_active_ratio, kappa):
    # Input: the network weight and the neuron activation ratio; Output: the theoretical capacity.
    assert weight.shape[0] == neuron_active_ratio.shape[0]
    n_neuron = weight.shape[0]
    initial_guess = [1, 1]
    # The number of connections equals the number of non-zero elements in the weight matrix:
    connection_matrix = (weight!=0).astype(int)
    in_degree = np.sum(connection_matrix, axis=1)
    # this implementation's running time is probably too long, optimize it later.
    capacity_set = np.zeros(len(neuron_active_ratio))
    for i in range(len(neuron_active_ratio)):
        perceptron_alpha = theoretical_perceptron_capacity([2*neuron_active_ratio[i]-1], kappa)
        # sol = root(equations_simplified, initial_guess, args=(2*neuron_active_ratio[i]-1), method='hybr')
        capacity_set[i] = perceptron_alpha[0]*in_degree[i]/n_neuron
        if capacity_set[i]<0:
            # failed to find the solution, set the capacity to infinity:
            capacity_set[i]=np.inf

    if np.min(capacity_set)<0:
        print("Warning: the capacity is negative!")

    return np.min(capacity_set)

def theoretical_network_capacity(weight, neuron_active_ratio):
    # Input: the network weight and the neuron activation ratio; Output: the theoretical capacity.
    assert weight.shape[0] == neuron_active_ratio.shape[0]
    n_neuron = weight.shape[0]
    initial_guess = [1, 1]
    # The number of connections equals the number of non-zero elements in the weight matrix:
    connection_matrix = (weight!=0).astype(int)
    in_degree = np.sum(connection_matrix, axis=1)
    # this implementation's running time is probably too long, optimize it later.
    capacity_set = np.zeros(len(neuron_active_ratio))
    for i in range(len(neuron_active_ratio)):
        sol = root(equations_simplified, initial_guess, args=(2*neuron_active_ratio[i]-1), method='hybr')
        capacity_set[i] = sol.x[0]*in_degree[i]/n_neuron
        if capacity_set[i]<0:
            # failed to find the solution, set the capacity to infinity:
            capacity_set[i]=np.inf

    if np.min(capacity_set)<0:
        print("Warning: the capacity is negative!")

    return np.min(capacity_set)

def theoretical_network_capacity_soft(weight, neuron_active_ratio, beta):
    # Input: the network weight and the neuron activation ratio; Output: the theoretical capacity.
    assert weight.shape[0] == neuron_active_ratio.shape[0]
    n_neuron = weight.shape[0]
    initial_guess = [1, 1]
    # The number of connections equals the number of non-zero elements in the weight matrix:
    connection_matrix = (weight!=0).astype(int)
    in_degree = np.sum(connection_matrix, axis=1)
    # this implementation's running time is probably too long, optimize it later.
    capacity_set = np.zeros(len(neuron_active_ratio))
    perceptron_alpha = np.zeros(len(neuron_active_ratio))
    for i in range(len(neuron_active_ratio)):
        sol = root(equations_simplified, initial_guess, args=(2*neuron_active_ratio[i]-1), method='hybr')
        capacity_set[i] = sol.x[0]*in_degree[i]/n_neuron
        if capacity_set[i]<0:
            # failed to find the solution, set the capacity to infinity:
            capacity_set[i]=np.inf

        #for debug:
        perceptron_alpha[i] = sol.x[0]

    if np.min(capacity_set)<0:
        print("Warning: the capacity is negative!")

    # average the capacities by the weight of softmin:
    capacity = np.sum(softmax(-beta*capacity_set)*capacity_set)

    return capacity