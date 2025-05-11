import numpy as np
from numpy import random
import torch
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from cmath import inf
import datetime
from sklearn import svm
from matplotlib.colors import Normalize
from sklearn.exceptions import ConvergenceWarning
import warnings
""" Author: Kaining Zhang, 05/04/2025 """

warnings.filterwarnings("ignore", category=ConvergenceWarning)
# training_max_iter = 100000

def make_pattern(n_pattern, n_neuron, perc_active = None, neuron_base_state = -1):
    # random.seed(1234)
    if perc_active is None:
        perc_active = torch.ones(n_neuron)*0.5
    elif isinstance(perc_active, float):
        perc_active = torch.ones(n_neuron)*perc_active
    patterns = torch.from_numpy(random.rand(n_pattern, n_neuron)).type(torch.float32)
    patterns[patterns>1-perc_active]=1-1e-5
    patterns[patterns<=1-perc_active]=neuron_base_state+1e-5
    return patterns


def retrieve_batch_patterns(patterns, network):
    # This function is slow in running speed, use network.evolve_batch instead.
    n_pattern = patterns.shape[0]
    n_neuron = patterns.shape[1]
    stored_patterns = torch.zeros((n_pattern, n_neuron))
    retrieval_time = torch.zeros(n_pattern)
    done = torch.zeros(n_pattern)
    for i in range(n_pattern):
        P = patterns[i]
        stored_patterns[i,:], done[i], retrieval_time[i] = network.evolve(P)
    return stored_patterns, done, retrieval_time

def L1_norm_dist(pattern1, pattern2):
    assert pattern1.shape[-1] == pattern2.shape[-1], "Two pattern's dimensions do not equal"
    assert pattern1.ndim<3 and pattern2.ndim<3, "Can't calculate patterns with dimension >=3"
    dist = np.linalg.norm(pattern1-pattern2, ord=1, axis=-1)
    return dist

def cosyne_similarity(pattern1, pattern2):
    assert pattern1.shape[-1] == pattern2.shape[-1], "Two pattern's dimensions do not equal"
    assert pattern1.ndim<3 and pattern2.ndim<3, "Can't calculate patterns with dimension >=3"
    if pattern1.ndim==1:
        pattern1 = pattern1[None,:] # add a dimension to the pattern
    if pattern2.ndim==1:
        pattern2 = pattern2[None,:]

    similarity = np.dot(pattern1, pattern2.T)/(np.linalg.norm(pattern1, axis=-1)[:,None]*np.linalg.norm(pattern2, axis=-1)[None,:])

    similarity = np.squeeze(similarity)
    similarity[similarity>1] = 1 # avoid numerical error
    similarity[similarity<-1] = -1
    return similarity

def calc_similarity_score(pattern1, pattern2):
    assert len(pattern1.shape) == 1 and len(pattern2.shape) == 1, "The input patterns should be 1D"
    N_act = np.sum(pattern1>0)
    N_act_to_act = np.sum((pattern1>0) & (pattern2>0))
    N_inact = np.sum(pattern1<0)
    N_inact_to_inact = np.sum((pattern1<0) & (pattern2<0))
    if N_act == 0:
        similarity = (N_inact_to_inact/N_inact)
    elif N_inact == 0:
        similarity = (N_act_to_act/N_act)
    else:
        similarity = ((N_act_to_act/N_act) + (N_inact_to_inact/N_inact))/2
    return similarity

def print_parameters_on_plot(ax, parameters:dict):
    text = ""
    for key in parameters.keys():
        text = text+key + '=' + str(parameters[key])+ "\n"
    ax.text(0.05, 0.05, text, transform=ax.transAxes, fontsize=7)
    ax.set_ylim([0,1])


def make_file_name(parameters:dict):
    file_name = '_'
    for key in parameters.keys():
        file_name = file_name + key + '=' + str(parameters[key]) + '_'
    return file_name[:-1]

def make_name_with_time(parameters:dict={}):
    file_name = '_'
    for key in parameters.keys():
        file_name = file_name + key + '=' + str(parameters[key]) + '_'
    return file_name[:-1] + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class Rescaled_Norm(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=0, alpha=10, clip=False):
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if value.size == 0:
            return value
        normalized = (value - self.vmin) / (self.vmax - self.vmin)
        normalized = np.log(10*normalized+1)
        normalized = (normalized-np.min(normalized))/(np.max(normalized)-np.min(normalized))
        return np.ma.masked_array(normalized)
    
# for making manual colormap
def custom_normalization(value, vmin, vmax):
    normalized = (value - vmin) / (vmax - vmin)  # Standard normalization
    normalized = np.log(10 * normalized + 1)     # Log transformation
    normalized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))  # Rescale
    return normalized

def create_custom_colormap(base_cmap, vmin, vmax, num_colors=256):
    values = np.linspace(vmin, vmax, num_colors)  # Generate values across range
    transformed_values = custom_normalization(values, vmin, vmax)  # Apply normalization
    # Get colors from base colormap
    base_colors = base_cmap(transformed_values)  # Map transformed values to colors
    # Create a new colormap
    new_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", base_colors, N=num_colors)
    return new_cmap


def concat_diagonal(matrix1, matrix2):
    # Get the dimensions of the input matrices
    rows1, cols1 = matrix1.size()
    rows2, cols2 = matrix2.size()

    # Calculate the size of the new matrix
    new_rows = max(rows1, rows2)
    new_cols = max(cols1, cols2)
    total_rows = new_rows + new_cols
    total_cols = new_rows + new_cols

    # Create a new matrix with appropriate dimensions
    new_matrix = torch.zeros(total_rows, total_cols)

    # Fill the diagonal blocks with the input matrices
    new_matrix[:rows1, :cols1] = matrix1
    new_matrix[rows1:, cols1:] = matrix2

    return new_matrix

def information_entropy(prob):
    # prob is a torch tensor describing the prob. distribution or a number describing the probability of a binary distribution
    if isinstance(prob, (int, float)):
        prob = torch.tensor([prob, 1-prob]) # binary distribution
    else:
        prob = prob/prob.sum() # normalize the probability!
    # calculate the entropy:
    entropy = -torch.sum(prob*torch.log2(prob))
    return entropy

def svm_score(W, b, patterns, label, kappa=0.0, regularization=None):
    # W: the weight matrix
    # patterns: the patterns to be learned
    # label: the label of the patterns
    # return: the accuracy of the SVM model
    # if isinstance(W, np.ndarray):
    #     W = torch.from_numpy(W)
    # if isinstance(b, np.ndarray):
    #     b = torch.from_numpy(b)
    # if isinstance(label, np.ndarray):
    #     label = torch.from_numpy(label)
    # if isinstance(patterns, np.ndarray):
    #     patterns = torch.from_numpy(patterns)
    n_pattern = patterns.shape[0]
    n_input = patterns.shape[1]
    assert label.shape[0] == n_pattern, "The number of labels should be equal to the number of patterns"
    # rescale W and b:
    if regularization is not None:
        normalizer = np.sqrt(n_input/np.sum(W**2*regularization))
        W = W*normalizer
        b = b*normalizer
    # calculate the accuracy:
    accuracy = 0
    for i in range(n_pattern):
        if (np.dot(W, patterns[i])+b)*label[i] > kappa*np.sqrt(n_input):
            accuracy += 1
    accuracy = accuracy/n_pattern
    return accuracy

def svm_by_sklearn(patterns, label, regularization=1.0, add_bias = True):
    n_neuron = patterns.shape[1]
    n_pattern = patterns.shape[0]
    if isinstance(regularization, (int, float)):
        regularization = np.ones(n_neuron)*regularization

    assert np.unique(label).shape[0] == 2, "The label should be binary"

    if add_bias:
        bias_const = 10*np.mean(1/regularization) # this number should be relatively large, but if it is too large, the classifer accurray also decrease. 
        scaled_patterns = patterns/np.sqrt(regularization)
        # append the bias term to the patterns
        scaled_patterns = np.concatenate((scaled_patterns, bias_const*np.ones((n_pattern, 1))), axis=1)

        clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, C=1.0, fit_intercept=False, class_weight=None, verbose=0, random_state=None, max_iter=1000)
        clf.fit(scaled_patterns, label)
        W = clf.coef_[0][:-1]/np.sqrt(regularization)
        b = clf.coef_[0][-1]*bias_const
    else:
        scaled_patterns = patterns/np.sqrt(regularization)
        clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, C=1.0, fit_intercept=False, class_weight=None, verbose=0, random_state=None, max_iter=1000)
        clf.fit(scaled_patterns, label)
        W = clf.coef_[0]/np.sqrt(regularization)
        b = 0
    # renormalize the weight:
    normalizer = np.sqrt(n_neuron/np.sum(regularization*W**2))
    W = W*normalizer
    b = b*normalizer
    
    return W, b

def network_score(W, b, patterns, kappa = 0.0):
    # The network score is the percentage of the unit of patterns whose state and the input aligns with each other.
    # W: the weight matrix
    # patterns: the patterns to be learned
    # return: the accuracy of the network model
    if isinstance(W, np.ndarray):
        W = torch.from_numpy(W)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)
    if isinstance(patterns, np.ndarray):
        patterns = torch.from_numpy(patterns)
    n_pattern = patterns.shape[0]
    n_neuron = patterns.shape[1]
    assert patterns.ndim == 2, "The patterns should be a 2D tensor"
    assert W.shape[0] == n_neuron, "The weight matrix should have the same number of neurons as the patterns"
    assert b.shape[0] == n_neuron, "The bias vector should have the same number of neurons as the patterns"

    in_degree = torch.sum(W!=0, dim=1)
    # calculate the accuracy:
    correct_matrix = (torch.matmul(patterns, W.T)+b)*patterns - kappa*np.sqrt(in_degree)[None,:]>0
    accuracy = torch.sum(correct_matrix)/n_pattern/n_neuron

    return accuracy
    