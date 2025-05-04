import numpy as np
from numpy import random
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import scipy
# from scipy.linalg import sqrtm
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
import utils_basic as utils
import math
from sklearn import svm
import sklearn


act_state = 1-1e-5
inact_state = -1+1e-5 # when the state_g is equal to 0 or 1, state_x will go to infinity
# training_max_iter = 100000



class random_connect_network(nn.Module):
    max_loop = 10000
    min_error = 2e-8

    def __init__(self, n_neuron, dt=0.01, network_type=None, connection_prop = None, W_symmetric=True, weight_std_init=None, mask=None, neuron_base_state = -1):
        # layer 1 is the feature layer, which receive and store the input.
        # layer 2 is the index layer.
        super().__init__()
        self.n_neuron = n_neuron
        self.dt = dt
        self.network_type = network_type
        self.lossfun = nn.MSELoss()
        self.W_symmetric = W_symmetric

        assert neuron_base_state in [0, -1] # make sure the neuron_base_state is either 0 or -1
        self.neuron_base_state = neuron_base_state
        self.act_state = 1-1e-5
        self.inact_state = neuron_base_state+1e-5 # when the state_g is equal to 0 or 1, state_x will go to infinity

        if weight_std_init is None:
            weight_std_init = 0.1
        self.weight_std_init = weight_std_init
        
        # create mask for certain weight of the network:
        if mask is None:
            # self.mask[i,j] = 1 means there is a connection from neuron j to neuron i, otherwise there is no connection.
            self.mask = torch.ones((self.n_neuron, self.n_neuron))
            self.mask-=torch.eye(self.n_neuron)
            # add heterogeneity to the network, such that some neurons have more connections than others:
            if connection_prop is None:
                connection_prop = 0.5*np.ones((self.n_neuron,))
            for i in range(self.n_neuron):
                self.mask[i] = self.mask[i] * torch.bernoulli(torch.ones(self.n_neuron) * connection_prop[i])
            self.mask = self.mask>0
        else:
            assert mask.shape == (self.n_neuron, self.n_neuron) # make sure the mask has the right shape.
            self.mask = mask>0

        # make sure the mask is symmetric:
        if self.W_symmetric:
            self.mask = (self.mask + self.mask.t())>0
        
        # initiate the weight matrix:
        self.weight = torch.normal(0, self.weight_std_init, (self.n_neuron, self.n_neuron))
        self.weight = self.weight * self.mask
        if self.W_symmetric:
            self.weight = (self.weight + self.weight.t())/2
        # initiate the threshold:
        self.b = torch.normal(0, self.weight_std_init, (self.n_neuron, ))
        
        self.tau = torch.ones(n_neuron)
        self.beta = torch.ones(n_neuron)*1e6 # Make it almost like a step function

        self.max_x = 500/self.beta.max().item()

    def reinitiate(self):
        self.weight = torch.normal(0, self.weight_std_init, (self.n_neuron, self.n_neuron))
        self.weight = self.weight * self.mask
        if self.W_symmetric:
            self.weight = (self.weight + self.weight.t())/2
        # initiate the threshold:
        self.b = torch.normal(0, self.weight_std_init, (self.n_neuron, ))       

    
    def g(self, x, beta): #activation_func
        if self.neuron_base_state == 0:
            return torch.sigmoid(beta*x)
        elif self.neuron_base_state == -1:
            return torch.tanh(beta*x)

    def g_inverse(self, g, beta): # inverse of g, clip at max_x
        if self.neuron_base_state == 0:
            state_x = (1/beta) * torch.log((1+g)/(1-g))
        elif self.neuron_base_state == -1:
            state_x = torch.arctanh(g)
        state_x = torch.clip(state_x, -self.max_x, self.max_x)
        return state_x

    def g_derivative(self, x, beta):
        if self.neuron_base_state == 0:
            g = self.g(x, beta)
            return beta*g*(1-g)
        elif self.neuron_base_state == -1:
            return beta*(1-self.g(x, beta)**2)
    
    def recurrence(self, state_x, state_g, dt):
        pass

    def forward(self, state_g, n_step = 1, dt_train = 1):
        # This forward function is specific for backprop training.
        state_x = self.g_inverse(state_g,self.beta)
        for _ in range(n_step):
            pre_activation = state_g@self.weight.T+self.b
            state_x = (dt_train/self.tau)*pre_activation + (1-dt_train/self.tau)*state_x
            state_g = self.g(state_x, self.beta)
        return state_g


    def train_svm(self, patterns, kappa=0, use_reg = True):
        if self.W_symmetric:
            raise Exception("This function only works for asymmetric network")
        
        for i in range(self.n_neuron):
            # clf = svm.LinearSVC(C=1, fit_intercept=True,dual=True)
            labels = torch.round(patterns[:,i]).to(torch.int)
            inputs = patterns[:, self.mask[i,:] == 1]
            n_input = inputs.shape[1]
            
            if use_reg:
                regularization = torch.var(inputs, axis=0)
                varmin = torch.var(inputs)*0.1+1e-3
                regularization[regularization<varmin] = varmin # prevent the regularization to be too small, which may cause numerical instability.
            else:
                regularization = torch.ones(n_input)

            if torch.all(labels!=1):
                # rescale the weight:
                W = self.weight[(i,),self.mask[i,:] == 1]
                normalizer = torch.sqrt(n_input/torch.sum(W**2*regularization))
                self.weight.data[i,self.mask[i,:] == 1] = W*normalizer
                # get the bias:
                self.b.data[i] = -torch.max(patterns@self.weight[(i,),:].T)-np.sqrt(n_input)*kappa-1e-1
            elif torch.all(labels==1):
                # rescale the weight:
                W = self.weight[(i,),self.mask[i,:] == 1]
                normalizer = torch.sqrt(n_input/torch.sum(W**2*regularization))
                self.weight.data[i,self.mask[i,:] == 1] = W*normalizer
                # get the bias:
                self.b.data[i] = -torch.min(patterns@self.weight[(i,),:].T)+np.sqrt(n_input)*kappa+1e-1
            elif inputs.shape[1]!=0:
                weights,b = utils.svm_by_sklearn(inputs.numpy(), labels.numpy(),regularization = regularization.numpy())
                self.weight.data[i,self.mask[i,:] == 1] = torch.from_numpy(weights).float()
                self.b.data[i] = b
                # clf.fit(inputs, labels)
                # self.weight.data[i,self.mask[i,:] == 1] = torch.from_numpy(clf.coef_[0]).float()
                # self.b.data[i] = torch.from_numpy(clf.intercept_).float()
            # else:
            #     print("Neuron i is not connected to any other neurons, and the labels are not all 1 or all 0.")


        stored_patterns = self.forward(patterns, n_step = 1, dt_train = 1)
        # network_accuracy = utils.network_score(self.weight.detach(), self.b.detach(), patterns, kappa = kappa)



        # Get the most acurate patterns that is stored in the network:
        if self.lossfun(stored_patterns, patterns)>self.min_error*2:
            success = False
        else:
            success = True

        return success, stored_patterns
    
    def train_PLA(self, patterns, lr=0.01, training_max_iter=10000):
        # Only apply to the case where kappa = 0
        n_pattern = patterns.shape[0]
        torch.set_grad_enabled(False)
        n_loop=0
        while  n_loop<=training_max_iter:
            check_array = torch.zeros(n_pattern).type(torch.bool)
            for i in range(n_pattern):
                P = patterns[i,:]
                P_sign = torch.sign(P-0.5) # the inactivation state is -1, and the activation state is 1
                
                x_p = (self.weight@P + self.b)*P_sign

                epsilon1 = torch.heaviside(-x_p + 1e-15 ,torch.tensor([1.0])) # plus the 1e-15 to prevent the computer calculation error.

                delta_W = lr*epsilon1[:,None]*P_sign[:,None]*P
                delta_b = lr*epsilon1*P_sign

                self.weight.data = self.weight+delta_W
                self.weight.data = self.weight-self.weight*torch.eye(self.n_neuron)
                self.b.data = self.b+delta_b

                # mask the weight matrix:
                self.weight.data = self.weight*self.mask
                # make sure the weight matrix is symmetric:
                if self.W_symmetric:
                    self.weight.data = (self.weight+self.weight.T)/2


                if torch.all(epsilon1==0):
                    check_array[i]=True
            
            if torch.all(check_array):
                break


            if n_loop%100==0:
                stored_patterns = self.forward(patterns, n_step = 1, dt_train = 1)
                loss = self.lossfun(patterns, stored_patterns)
                print(f'{n_loop} of loops done, out of {training_max_iter}')
                print(f'loss: {loss.item()}')
            
            n_loop+=1
        if n_loop>=training_max_iter:
            success = False
        else:
            success = True

        stored_patterns = self.forward(patterns, n_step = 1, dt_train = 1)

        # Get the most acurate patterns that is stored in the network:
        if self.lossfun(stored_patterns, patterns) > self.min_error*2:
            success = False

        torch.set_grad_enabled(True)

        return success, stored_patterns



    
        
        
        