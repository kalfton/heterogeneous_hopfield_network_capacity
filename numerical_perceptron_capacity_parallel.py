import numpy as np
from numpy import random
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import scipy
from scipy.linalg import sqrtm
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
import utils_basic as utils
import utils_theoretical
import math
from sklearn import svm
from scipy.integrate import quad
from scipy.optimize import root
import time
import argparse
import multiprocessing as mp
import sys

# TODO: Make the for loop parallel

# create data points for svm
def create_data_normal(n_data, n_neuron, ratio_label = 0.5, mean=1.0, var=1, correlation = 0.0):
    # x = np.random.multivariate_normal(mean*np.ones(n_neuron), var*np.eye(n_neuron)+correlation - correlation*np.eye(n_neuron), n_data)
    # an equivalent way to generate correlated gaussian data upto a linear transformation
    corr_matrix = np.eye(n_neuron)*(var-correlation)
    corr_matrix[0,0] = var+(n_neuron-1)*correlation
    x = np.random.multivariate_normal(mean*np.ones(n_neuron), corr_matrix, n_data)
    y = np.random.rand(n_data)
    y[y > 1-ratio_label] = 1
    y[y<= 1-ratio_label] = -1

    return x, y

# create_data_normal(100, 101, ratio_label = 0.5, mean=0, var=1, correlation = 0.99)

def create_data_normal_correlated_bypattern(n_data, n_neuron, ratio_label = 0.5, mean=0, var=1, correlation = 0.0):
    x = np.random.multivariate_normal(mean*np.ones(n_data), var*np.eye(n_data)+correlation - correlation*np.eye(n_data), n_neuron).T
    y = np.random.rand(n_data)
    y[y > 1-ratio_label] = 1
    y[y<= 1-ratio_label] = -1
    return x, y


def create_data_gaussian_markov_chain(n_data, n_neuron, ratio_label = 0.5, phi=0.5, sigma=1.0, bias=0.0):
    # phi is the correlation coefficient between the current state and the next state
    
    # Initialize output array
    sequences = np.zeros((n_data, n_neuron))

    one_minus_phi = np.sqrt(1 - phi**2)
    
    for sample_idx in range(n_data):
        # Starting value
        x_i = np.random.normal(0, sigma)
        
        for seq_idx in range(n_neuron):
            # Gaussian noise
            epsilon_i = np.random.normal(0, sigma)
            
            # Markov chain formula
            x_i = phi * x_i + one_minus_phi*epsilon_i
            
            sequences[sample_idx, seq_idx] = x_i

    sequences = sequences+bias

    y = np.random.rand(n_data)
    y[y > 1-ratio_label] = 1
    y[y<= 1-ratio_label] = -1
            
    return sequences, y

def create_data_exponential(n_data, n_neuron, ratio_label = 0.5, mean=1.0, bias=0.0):
    x = np.random.exponential(mean, (n_data, n_neuron))+bias
    y = np.random.rand(n_data)
    y[y > 1-ratio_label] = 1
    y[y<= 1-ratio_label] = -1

    return x, y

def create_data_poisson(n_data, n_neuron, ratio_label = 0.5, mean=1.0, bias=0.0):
    x = np.random.poisson(mean, (n_data, n_neuron))+bias
    y = np.random.rand(n_data)
    y[y > 1-ratio_label] = 1
    y[y<= 1-ratio_label] = -1

    return x, y

def create_data_binary(n_data, n_neuron, ratio_active = 0.5, ratio_label = 0.5):
    x = np.random.rand(n_data, n_neuron)
    x[x > 1-ratio_active] = 1
    x[x<= 1-ratio_active] = -1
    y = np.random.rand(n_data)
    y[y > 1-ratio_label] = 1
    y[y<= 1-ratio_label] = -1
    return x, y

def create_data_mixed(n_data, n_neuron, pars1 = 0.5, ratio_label = 0.5, bias = 0.0):
    x1 = np.random.rand(n_data, n_neuron//4)
    x1[x1 > 1-pars1] = 1
    x1[x1<= 1-pars1] = -1

    x2 = np.random.poisson(pars1, (n_data, n_neuron//4))
    x3 = np.random.exponential(pars1, (n_data, n_neuron//4))
    x4 = np.random.normal(pars1,  scale=1.0, size=(n_data, (n_neuron - 3*x1.shape[1])))

    x = np.concatenate((x1, x2, x3, x4), axis=1)

    y = np.random.rand(n_data)
    y[y > 1-ratio_label] = 1
    y[y<= 1-ratio_label] = -1
    return x, y

def create_data_binary_from_uniform(n_data, n_neuron, ratio_active_mean = 0.5, ratio_active_range = 0.1, ratio_label = 0.5):

    ratio_active = np.ones((n_data,1))@ ((np.random.rand(1,n_neuron)-0.5)*ratio_active_range + ratio_active_mean)
    x = np.random.rand(n_data, n_neuron)
    x[x > 1-ratio_active] = 1
    x[x<= 1-ratio_active] = -1
    y = np.random.rand(n_data)
    y[y > 1-ratio_label] = 1
    y[y<= 1-ratio_label] = -1
    return x, y

def create_data_binary_multiratio(n_data, n_neuron, ratio_active1 = 0.5, ratio_active2 = 0.5, ratio_label = 0.5):
    
    x1 = np.random.rand(n_data, int(np.floor(n_neuron/5)))
    x1[x1 > 1-ratio_active1] = 1
    x1[x1<= 1-ratio_active1] = -1

    x2 = np.random.rand(n_data, int(n_neuron-np.floor(n_neuron/5)))
    x2[x2 > 1-ratio_active2] = 1
    x2[x2<= 1-ratio_active2] = -1

    x = np.concatenate((x1, x2), axis=1)

    y = np.random.rand(n_data)
    y[y > 1-ratio_label] = 1
    y[y<= 1-ratio_label] = -1
    return x, y

def binary_search_capacity(n_neuron, max_n_data, method, ratio_label = 0.5, add_bias = True, epsilon = 1e-5):
    # binary search for the capacity:
    left = 0
    right = max_n_data
    while left <= right:
        mid = (left + right) // 2
        x_data, label = method(mid, n_neuron, ratio_label = ratio_label)
        if np.unique(label).shape[0] == 1:
            left = mid + step
        else:
            # svm by sklearn
            # clf = svm.LinearSVC(C=1, fit_intercept=add_bias)
            # score = clf.score(x_data, label)

            # manual svm:
            regularization = np.var(x_data,axis=0)
            varmin = np.var(x_data)*0.1
            regularization[regularization<varmin] = varmin # prevent the regularization to be too small, which may cause numerical instability
            W, b = utils.svm_by_sklearn(x_data, label, regularization=regularization, add_bias=add_bias)
            # W, b = utils.svm_learning_alg(x_data, label, C = 100)
            score = utils.svm_score(W, b, x_data, label, kappa = kappa, regularization = regularization)

            if score>1-epsilon:
                left = mid + step
            else:
                right = mid - step
    return left
            

def capacity_for_parallel(par1, par2, syspars, step, method="binary"):
    epsilon = syspars["epsilon"]
    n_neuron = syspars["n_neuron"]
    n_repeat = syspars["n_repeat"]
    add_bias =  syspars["add_bias"]
    kappa = syspars["kappa"]
    use_reg = syspars["use_kappa"]

    capacity = np.zeros(n_repeat)
    for k in range(n_repeat):
        # binary search for the capacity:
        left = syspars["min_capacity"]
        right = syspars["max_capacity"]
        while left <= right:
            mid = (left + right) // 2
            if method == "binary":
                x_data, label = create_data_binary(mid, n_neuron, ratio_active = par1, ratio_label = par2)
            elif method == "normal":
                x_data, label = create_data_normal(mid, n_neuron, ratio_label = par2, var=1.0, mean=par1, correlation= 0)
            elif method == "markov":
                x_data, label = create_data_gaussian_markov_chain(mid, n_neuron, ratio_label = par2, phi = par1, sigma=1.0, bias=0.0)

            if np.unique(label).shape[0] == 1:
                left = mid + step
            else:
                # svm by sklearn
                # clf = svm.LinearSVC(C=1, fit_intercept=add_bias)
                # clf.fit(x_data, label)
                # score = clf.score(x_data, label)

                # manual svm:
                if use_reg:
                    regularization = np.var(x_data, axis=0)
                    varmin = np.var(x_data)*0.1+1e-3
                    regularization[regularization<varmin] = varmin # prevent the regularization to be too small, which may cause numerical instability.
                else:
                    regularization = np.ones(n_neuron)

                # regularization = np.var(x_data,axis=0)
                # regularization[regularization<1e-5] = 1e-5 # avoid numerical instability
                W, b = utils.svm_by_sklearn(x_data, label, regularization=regularization, add_bias=add_bias)
                # W, b = utils.svm_learning_alg(x_data, label, C = 100)
                score = utils.svm_score(W, b, x_data, label, kappa = kappa)

                if score>1-epsilon:
                    left = mid + step
                else:
                    right = mid - step

        capacity[k] = left

    return capacity

if __name__ == '__main__':

    # sys.argv += "--n_neuron 500 --add_bias --n_repeat 2".split()

    parser = argparse.ArgumentParser(description='Calculate the capacity of perceptron for different distributions')
    parser.add_argument('--n_neuron', type=int, default=100, help='number of neurons')
    parser.add_argument('--n_repeat', type=int, default=10, help='number of repeats')
    parser.add_argument('--add_bias', action='store_true', help='add bias term or not')
    parser.add_argument('--epsilon', type=float, default=1e-2, help='accuracy of the capacity')
    parser.add_argument('--kappa', type=float, default=0.5, help='kappa value for the perceptron')
    parser.add_argument('--use_kappa', action='store_true', help='use kappa or not')
    parser.add_argument('--n_process', type=int, default=10, help='number of processes to run in parallel')

    pars = vars(parser.parse_args())

    epsilon = pars["epsilon"]
    n_neuron = pars["n_neuron"]
    n_repeat = pars["n_repeat"]
    add_bias =  pars["add_bias"]
    if not pars["use_kappa"]:
        pars["kappa"]=0.0
    kappa = pars["kappa"]
    n_process = np.min((pars["n_process"], mp.cpu_count()))

    # Binary distribution
    # create data points for svm
    # input_mean_set = -1+2*np.concatenate((np.arange(0.01,0.09, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.999, 0.03)))
    # output_mean_set = -1+2*np.concatenate((np.arange(0.01,0.09, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.999, 0.03)))
    input_mean_set = -1+2*np.arange(0.01,0.9999, 0.02)
    output_mean_set = -1+2*np.arange(0.01,0.9999, 0.02)

    step = int(n_neuron/10)
    n_data_set = np.arange(step,7*n_neuron, step)
    pars["max_capacity"] = n_data_set[-1]
    pars["min_capacity"] = n_data_set[0]
    max_capacity = pars["max_capacity"]
    min_capacity = pars["min_capacity"]


    capacities = [[[] for _ in range(len(output_mean_set))] for _ in range(len(input_mean_set))]
    pool_results_object = [[[] for _ in range(len(output_mean_set))] for _ in range(len(input_mean_set))]

    with mp.Pool(processes=n_process) as pool:
        for i in range(len(input_mean_set)):
            ratio_active = (input_mean_set[i]+1)/2
            for j in range(len(output_mean_set)):
                ratio_label = (output_mean_set[j]+1)/2
                pool_results_object[i][j] = pool.apply_async(capacity_for_parallel, args=(ratio_active, ratio_label, pars, step, "binary"))
                print("ratio_active: ", ratio_active, "ratio_label: ", ratio_label)
                # for debug:
                # capacities[i][j] = capacity_for_parallel(ratio_active, ratio_label, pars, step, "binary")

        # get the results
        for i in range(len(input_mean_set)):
            for j in range(len(output_mean_set)):
                capacities[i][j] = pool_results_object[i][j].get()


    capacities = np.array(capacities)
    capacities = np.mean(capacities, axis=2)
    capacities = capacities/n_neuron

    # plot the capacity:
    plt.figure()
    # plot the meshgrid of capacity matrix:
    xx, yy = np.meshgrid(output_mean_set, input_mean_set)
    h = plt.contourf(xx, yy, capacities, cmap='viridis', vmin=0, vmax=7, levels = np.arange(0, 7.1, 0.5))
    plt.axis('scaled')
    # label the colorbar:
    cbar = plt.colorbar()
    cbar.set_label("capacity", rotation=270, labelpad=20)
    plt.xlabel("m_out")
    plt.ylabel("m_in")
    plt.title("kappa = {0}".format(kappa))
    # make the plot square:
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show(block=False)
    plt.savefig("results/perceptron_capacity_binary_bias_n="+ str(n_neuron) +"k =" + str(kappa) + utils.make_name_with_time()+ ".pdf")

    # Correlated Gaussian distribution:
    input_corr_set = np.concatenate((np.arange(0.0, 0.9, 0.1), np.arange(0.9, 0.999, 0.03)))
    # input_mean_set = -1+2*np.concatenate((np.arange(0.00,0.09, 0.05), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.999, 0.03)))
    output_mean_set = -1+2*np.concatenate((np.arange(0.01,0.09, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.999, 0.03)))
    # ratio_label_set = np.concatenate((np.arange(0.04,0.09, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.97, 0.03)))


    capacities = [[[] for _ in range(len(output_mean_set))] for _ in range(len(input_corr_set))]
    pool_results_object = [[[] for _ in range(len(output_mean_set))] for _ in range(len(input_corr_set))]

    with mp.Pool(processes=n_process) as pool:
        for i in range(len(input_corr_set)):
            input_corr = input_corr_set[i]
            for j in range(len(output_mean_set)):
                ratio_label = (output_mean_set[j]+1)/2
                pool_results_object[i][j] = pool.apply_async(capacity_for_parallel, args=(input_corr, ratio_label, pars, step, "markov"))
                print("input_corr: ", input_corr, "ratio_label: ", ratio_label)
                # for debug:
                # capacities[i][j] = capacity_for_parallel(input_corr, ratio_label, pars, step, "markov")

        # get the results
        for i in range(len(input_corr_set)):
            for j in range(len(output_mean_set)):
                capacities[i][j] = pool_results_object[i][j].get()


    capacities = np.array(capacities)
    capacities = np.mean(capacities, axis=2)
    capacities = capacities/n_neuron

    # plot the capacity:
    plt.figure()
    # plot the meshgrid of capacity matrix:
    xx, yy = np.meshgrid(output_mean_set, input_corr_set)
    h = plt.contourf(xx, yy, capacities, cmap='viridis', vmin=0, vmax=7, levels = np.arange(0, 7.1, 0.5))
    plt.axis('scaled')
    # label the colorbar:
    cbar = plt.colorbar()
    cbar.set_label("capacity", rotation=270, labelpad=20)
    plt.xlabel("m_out")
    plt.ylabel("corr coeff")
    # make the plot square:
    plt.gca().set_aspect(2)
    plt.show(block=False)
    plt.savefig("results/perceptron_capacity_correlated_gaussian_n="+ str(n_neuron) + utils.make_name_with_time()+ ".pdf")


    # compare the capacity of svm for different distributions:
    output_mean_set = -1+2*np.concatenate((np.arange(0.04,0.09, 0.03), np.arange(0.1, 0.9, 0.1), np.arange(0.9, 0.97, 0.03)))

    step = int(n_neuron/10)
    # n_data_set = np.arange(step,7*n_neuron, step)
    max_capacity = 7*n_neuron
    min_capacity = 0


    capacities_binary = np.zeros((len(output_mean_set), n_repeat))+max_capacity
    capacities_normal = np.zeros((len(output_mean_set), n_repeat))+max_capacity
    capacities_markov = np.zeros((len(output_mean_set), n_repeat))+max_capacity
    capacities_exponential = np.zeros((len(output_mean_set), n_repeat))+max_capacity
    capacities_poisson = np.zeros((len(output_mean_set), n_repeat))+max_capacity
    capacities_mixed = np.zeros((len(output_mean_set), n_repeat))+max_capacity


    # binary distribution:
    for i in range(len(output_mean_set)):
        # binary distribution:
        ratio_active = 0.5
        ratio_label = (output_mean_set[i]+1)/2
        capacity = np.zeros(n_repeat)
        for k in range(n_repeat):
        #binary search for the capacity:
            capacity[k] = binary_search_capacity(n_neuron, max_capacity, create_data_binary, ratio_label = ratio_label, add_bias = add_bias, epsilon = epsilon)

        capacities_binary[i,:] = capacity.copy()
    capacities_binary = capacities_binary/n_neuron

    # normal distribution:
    for i in range(len(output_mean_set)):
        # normal distribution:
        ratio_active = 0.5
        ratio_label = (output_mean_set[i]+1)/2
        capacity = np.zeros(n_repeat)
        for k in range(n_repeat):
        #binary search for the capacity:
            capacity[k] = binary_search_capacity(n_neuron, max_capacity, create_data_normal, ratio_label = ratio_label, add_bias = add_bias, epsilon = epsilon)

        capacities_normal[i,:] = capacity.copy()
    capacities_normal = capacities_normal/n_neuron

    # markov distribution:
    for i in range(len(output_mean_set)):
        # markov distribution:
        ratio_active = 0.5
        ratio_label = (output_mean_set[i]+1)/2
        capacity = np.zeros(n_repeat)
        for k in range(n_repeat):
        #binary search for the capacity:
            capacity[k] = binary_search_capacity(n_neuron, max_capacity, create_data_gaussian_markov_chain, ratio_label = ratio_label, add_bias = add_bias, epsilon = epsilon)

        capacities_markov[i,:] = capacity.copy()
    capacities_markov = capacities_markov/n_neuron

    # exponential distribution:
    for i in range(len(output_mean_set)):
        # exponential distribution:
        ratio_active = 0.5
        ratio_label = (output_mean_set[i]+1)/2
        capacity = np.zeros(n_repeat)
        for k in range(n_repeat):
        #binary search for the capacity:
            capacity[k] = binary_search_capacity(n_neuron, max_capacity, create_data_exponential, ratio_label = ratio_label, add_bias = add_bias, epsilon = epsilon)

        capacities_exponential[i,:] = capacity.copy()
    capacities_exponential = capacities_exponential/n_neuron

    # poisson distribution:
    for i in range(len(output_mean_set)):
        # poisson distribution:
        ratio_active = 0.5
        ratio_label = (output_mean_set[i]+1)/2
        capacity = np.zeros(n_repeat)
        for k in range(n_repeat):
        #binary search for the capacity:
            capacity[k] = binary_search_capacity(n_neuron, max_capacity, create_data_poisson, ratio_label = ratio_label, add_bias = add_bias, epsilon = epsilon)

        capacities_poisson[i,:] = capacity.copy()
    capacities_poisson = capacities_poisson/n_neuron


    # mixed distribution:
    for i in range(len(output_mean_set)):
        # mixed distribution:
        ratio_active = 0.5
        ratio_label = (output_mean_set[i]+1)/2
        capacity = np.zeros(n_repeat)
        for k in range(n_repeat):
        #binary search for the capacity:
            capacity[k] = binary_search_capacity(n_neuron, max_capacity, create_data_mixed, ratio_label = ratio_label, add_bias = add_bias, epsilon = epsilon)

        capacities_mixed[i,:] = capacity.copy()
    capacities_mixed = capacities_mixed/n_neuron

    # calculate the theoretical capacity:
    capacities_theoretical = utils_theoretical.theoretical_perceptron_capacity(output_mean_set, kappa)


    plt.figure()
    # plot the mean and error bar:
    plt.errorbar(output_mean_set, capacities_binary.mean(axis=1), yerr=capacities_binary.std(axis=1), label = "binary")
    plt.errorbar(output_mean_set, capacities_normal.mean(axis=1), yerr=capacities_normal.std(axis=1), label = "normal")
    plt.errorbar(output_mean_set, capacities_markov.mean(axis=1), yerr=capacities_markov.std(axis=1), label = "markov")
    plt.errorbar(output_mean_set, capacities_exponential.mean(axis=1), yerr=capacities_exponential.std(axis=1), label = "exponential")
    plt.errorbar(output_mean_set, capacities_poisson.mean(axis=1), yerr=capacities_poisson.std(axis=1), label = "poisson")
    plt.errorbar(output_mean_set, capacities_mixed.mean(axis=1), yerr=capacities_mixed.std(axis=1), label = "mixed")
    plt.plot(output_mean_set, capacities_theoretical, label = "theoretical")


    plt.xlabel("m_out")
    plt.ylabel("alpha_c")
    plt.title("kappa = {0}".format(kappa))
    plt.ylim(0, 7.5)
    plt.legend()
    plt.show(block=False)
    # plt.savefig("results/comparison_of_capacity_with_different_distributions_n=100.png")
    plt.savefig("results/comparison_of_capacity_with_different_distributions_n="+str(n_neuron) +"k =" + str(kappa) +utils.make_name_with_time()+".pdf")


