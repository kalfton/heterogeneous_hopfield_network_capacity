import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
import utils_basic as utils
import math
""" Author: Kaining Zhang, 05/04/2025 """


class committee_machine(nn.Module):
    """A three layer neural network with a tree-like structure,
    The first layer is the input, the second layer is the hidden layer, and the third layer is the output containing only one unit.
    The connection between the input and the hidden layer is random and can be trained, and the connection between the hidden layer and the output layer is fixed as 1. 
    """
    def __init__(self, input_size, hidden_size, n_synapse_array=None, weight_std_init=0.01, add_bias=False):
        """ n_synapse_array is the number of synapses that each neuron in the hidden layer receives from the input layer.
        It should be a 1D array with the length of hidden_size."""
        super(committee_machine, self).__init__()
        # self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_std_init = weight_std_init
        self.add_bias = add_bias

        if n_synapse_array is None:
            n_synapse_per_neuron = input_size // hidden_size
            n_synapse_array = np.ones((hidden_size,)) * n_synapse_per_neuron
            n_synapse_array[-1] = n_synapse_array[-1] + input_size - np.sum(n_synapse_array) # make sure the total number of synapses is equal to the input size


        assert input_size == torch.sum(n_synapse_array) # make sure the input size is equal to the sum of the number of synapses that each neuron in the hidden layer receives from the input layer.

        self.n_synapse_array = n_synapse_array
        # Define trainable weights for connections between input and hidden layers
        # It should be a list containing the weights of each synapse from the input layer to the hidden layer.
        self.input_to_hidden = [torch.randn(n_synapse_array[i].item())*weight_std_init for i in range(hidden_size)]
        input_to_hidden_tensor = self.pad_list_of_tensors(self.input_to_hidden)
        # self.input_to_hidden = nn.ParameterList([nn.Parameter(torch.randn(n_synapse_array[i].item())*weight_std_init, requires_grad=True) for i in range(hidden_size)])
        self.input_to_hidden_tensor = nn.Parameter(input_to_hidden_tensor, requires_grad=True)
        if add_bias:
            self.hidden_biases = nn.Parameter(torch.randn(hidden_size)*weight_std_init, requires_grad=True)
        else:
            self.hidden_biases = torch.zeros(hidden_size, requires_grad=False)

        # Fixed weights for connections between hidden layer and output layer
        self.hidden_to_output = torch.ones(hidden_size) # important! change it back to not trainable # nn.Parameter(torch.ones(hidden_size),  requires_grad= True) #
        self.beta = 1.0 # the parameter for the activation function

    def reinitiate(self):
        self.input_to_hidden = [torch.randn(self.n_synapse_array[i].item())*self.weight_std_init for i in range(self.hidden_size)]
        self.input_to_hidden_tensor = nn.Parameter(self.pad_list_of_tensors(self.input_to_hidden), requires_grad=True)
        if self.hidden_biases.requires_grad:
            self.hidden_biases = nn.Parameter(torch.randn(self.hidden_size)*self.weight_std_init, requires_grad=True)
        else:
            self.hidden_biases = torch.zeros(self.hidden_size, requires_grad=False)
        # self.hidden_to_output = nn.Parameter(torch.ones(self.hidden_size), requires_grad=False)

    def align_weights(self):
        """Align the input_to_hidden weights to be the same as the input_to_hidden_tensor weights."""
        self.input_to_hidden = self.unpad_list_of_tensors(self.input_to_hidden_tensor, self.n_synapse_array)

    def forward_discrete(self, inputs):
        # use the sign function as the activation function
        # input should be a list of input vectors, each of which is a 1D tensor.
        hidden_fields = torch.zeros(self.hidden_size)
        for i in range(self.hidden_size):
            hidden_fields[i] = torch.sum(inputs[i] * self.input_to_hidden[i]) + self.hidden_biases[i]

        hidden_states = torch.sign(hidden_fields)
        output_field = torch.sum(hidden_states * self.hidden_to_output)
        output = torch.sign(output_field)

        return output, output_field, hidden_fields
    
    def forward_discrete_tensor_version(self, input_to_hidden_tensor, inputs):
        # use the sign function as the activation function
        # input should be a 3 dimensional tensor with the shape of (n_pattern, n_hidden, :)
        hidden_fields  = torch.einsum('ij,kij->ki', input_to_hidden_tensor, inputs)+ self.hidden_biases
        hidden_states = torch.sign(hidden_fields)
        output_field = torch.sum(hidden_states * self.hidden_to_output, dim=1)
        output = torch.sign(output_field)

        return output, output_field, hidden_fields

    def forward_tensor_version(self, inputs):
        """Forward pass through the network for backprop method. inputs should be a list of input vectors, each of which is a 1D tensor."""
        # input should be a 3 dimensional tensor with the shape of (n_pattern, n_hidden, :)
        hidden_fields  = torch.einsum('ij,kij->ki', self.input_to_hidden_tensor, inputs)+ self.hidden_biases
        hidden_states = self.activation_func(hidden_fields/torch.sqrt(self.n_synapse_array))
        output_field = torch.sum(hidden_states * self.hidden_to_output, dim=1)
        output = self.activation_func(output_field/math.sqrt(self.hidden_size))

        return output, output_field, hidden_fields


    def activation_func(self, x):
        #return torch.sign(x)
        return torch.tanh(self.beta*x)
    
    def pad_list_of_tensors(self, tensor_list):
        """
        tensor_list: list of 1D torch.Tensors of varying lengths
        return: 2D torch.Tensor [num_tensors, max_length]
        """
        max_len = max(t.size(0) for t in tensor_list)
        padded = torch.zeros((len(tensor_list), max_len), dtype=tensor_list[0].dtype)
        
        # Copy each tensor into the rows of 'padded' (fill the remainder with zeros).
        for i, t in enumerate(tensor_list):
            padded[i, :t.size(0)] = t
        
        return padded
    
    def unpad_list_of_tensors(self, padded_tensor, lengths):
        """
        Inverse of 'pad_list_of_tensors':
        - padded_tensor: a 2D torch.Tensor of shape [N, max_len]
        - lengths: a list of integers specifying the true length of each row
        Returns:
        - a list of 1D tensors of the specified lengths
        """
        output = []
        for i, length in enumerate(lengths):
            # Take only the first `length` elements from row i
            output.append(padded_tensor[i, :length])
        return output
    

    def train_backprop(self, inputs, labels, learning_rate=0.01, max_epochs=10000):
        """Train the network with given inputs and labels."""
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()
        n_pattern = len(inputs)
        inputs_tensor = []
        for i in range(n_pattern):
            inputs_tensor.append(self.pad_list_of_tensors(inputs[i]))
        input_tensor = torch.stack(inputs_tensor, dim=0)
        

        errors_for_return = []
        epochs_for_return = []
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            output_all, out_put_field_all, hidden_fields_all = self.forward_tensor_version(input_tensor)
            loss = torch.sum((output_all-labels)**2)
            loss = loss/n_pattern
            loss.backward()
            optimizer.step()

            if loss.item()<1e-8:
                # unpad the tensor
                self.align_weights()
                return True, errors_for_return, epochs_for_return

            if epoch%10==0:
                errors_for_return.append(torch.mean((output_all*labels<=0).float()).item())
                epochs_for_return.append(epoch)
                # control the absolute value of the weights:
                for i in range(self.hidden_size):
                    radius_square = torch.sum(self.input_to_hidden_tensor[i]**2)
                    if radius_square > self.n_synapse_array[i]:
                        with torch.no_grad():
                            self.input_to_hidden_tensor[i] = self.input_to_hidden_tensor[i]*math.sqrt(self.n_synapse_array[i]/radius_square)

            
        # print('Loss = {}'.format(loss.item()))
        self.align_weights()

        return False, errors_for_return, epochs_for_return
    

    # def train_ALA(self, inputs, labels, learning_rate=0.1, max_epochs=10000):
    #     """Train the network with given inputs and labels useing ALA learning.
    #     input should be a list of list of input vectors, each of which is a 1D tensor.
    #     label should be a one dim array""" 
    #     # ALA only works for no bias case
    #     self.add_bias = False
    #     self.hidden_biases = torch.zeros(self.hidden_size, requires_grad=False)
    #     torch.set_grad_enabled(False)
    #     n_pattern = len(inputs)
    #     errors_for_return = []
    #     epochs_for_return = []

    #     inputs_tensor = []
    #     for i in range(n_pattern):
    #         inputs_tensor.append(self.pad_list_of_tensors(inputs[i]))
    #     input_tensor = torch.stack(inputs_tensor, dim=0)
    #     input_to_hidden_tensor = self.pad_list_of_tensors(self.input_to_hidden)

    #     for epoch in range(max_epochs):
    #         output_fields_correction = torch.zeros(n_pattern)
    #         output_all = torch.zeros(n_pattern)

    #         output_all, out_put_field_all, hidden_fields_all = self.forward_discrete_tensor_version(input_to_hidden_tensor, input_tensor)
    #         output_fields_correction = out_put_field_all*labels

            
    #         if epoch%10==0:
    #             # calculate the error and save it
    #             errors_for_return.append(torch.mean((output_all*labels<=0).float()).item())
    #             epochs_for_return.append(epoch)

    #         # find the pattern that has the most negative output field 
    #         if torch.all(output_fields_correction>0):
    #             # unpad the tensor
    #             self.input_to_hidden_tensor = nn.Parameter(input_to_hidden_tensor, requires_grad=True)
    #             self.align_weights()
    #             torch.set_grad_enabled(True)
    #             return True, errors_for_return, epochs_for_return
    #         idx = torch.argmin(output_fields_correction)
    #         # find the hidden unit that is the easiest to change to correct the output
    #         hidden_fields = (self.forward_discrete_tensor_version(input_to_hidden_tensor, inputs_tensor[idx].unsqueeze(0))[2])[0]
    #         hidden_fields_correction = hidden_fields*labels[idx]
    #         hidden_fields_correction_neg = hidden_fields_correction
    #         hidden_fields_correction_neg[hidden_fields_correction>0] = -float('inf')
    #         idx_hidden = torch.argmax(hidden_fields_correction_neg)
    #         # update the weights
    #         input_to_hidden_tensor[idx_hidden] = input_to_hidden_tensor[idx_hidden] + learning_rate*inputs_tensor[idx][idx_hidden]*labels[idx] *(1-hidden_fields_correction[idx_hidden])
        

    #     # unpad the tensor
    #     self.input_to_hidden_tensor = nn.Parameter(input_to_hidden_tensor, requires_grad=True)
    #     self.align_weights()

    #     torch.set_grad_enabled(True)
    #     return False, errors_for_return, epochs_for_return

    

class random_connect_network(nn.Module):
    # max_loop = 10000
    min_error = 2e-8

    def __init__(self, n_neuron, dt=0.01, connection_prop = None, hidden_size = 5, weight_std_init=0.1, mask=None, neuron_base_state = -1, add_bias=False):
        # layer 1 is the feature layer, which receive and store the input.
        # layer 2 is the index layer.
        super().__init__()
        self.n_neuron = n_neuron
        self.dt = dt
        self.lossfun = nn.MSELoss()
        self.hidden_size = hidden_size # hidden_size is the number of hidden units in committee machine
        self.add_bias = add_bias

        assert neuron_base_state in [0, -1] # make sure the neuron_base_state is either 0 or -1


        self.weight_std_init = weight_std_init
        
        # create mask for certain weight of the network:
        if mask is None:
            # self.mask[i,j] = 1 means there is a connection from neuron j to neuron i, otherwise there is no connection.
            self.mask = torch.ones((self.n_neuron, self.n_neuron))
            self.mask-=torch.eye(self.n_neuron)
            # add heterogeneity to the network, such that some neurons have more connections than others:
            if connection_prop is None:
                connection_prop = 0.5*torch.ones((self.n_neuron,))
            for i in range(self.n_neuron):
                self.mask[i] = self.mask[i] * torch.bernoulli(torch.ones(self.n_neuron) * connection_prop[i])
            self.mask = self.mask>0
        else:
            assert mask.shape == (self.n_neuron, self.n_neuron) # make sure the mask has the right shape.
            self.mask = mask>0

        #self.weight = torch.normal(0, self.weight_std_init, (self.n_neuron, self.n_neuron)) # the synaptic weight from neuron to dendrites

        # create an matrix showing which dendrate each neuron's output is reached to
        self.dendrate = torch.randint(0, hidden_size, (self.n_neuron, self.n_neuron))
        self.dendrate[self.mask==0] = -1

        # Each neuron in the network is a instance of committee_machine:
        self.idx_in_dendrate = -torch.ones((self.n_neuron, self.n_neuron)).to(torch.int32) 
        # the index of within the dendrate (hidden unit) that each neuron is reached to, if equal to -1, then it is not reached to any dendrate.
        self.neuron_list = []
        for i in range(self.n_neuron):
            n_synapse_array = torch.zeros((hidden_size,)).to(torch.int32)
            for j in range(hidden_size):
                n_synapse_array[j] = torch.sum((self.dendrate[i]==j) & self.mask[i])
                self.idx_in_dendrate[i, (self.dendrate[i]==j) & self.mask[i]] = torch.arange(torch.sum((self.dendrate[i]==j) & self.mask[i])).to(torch.int32)
            input_size = torch.sum(n_synapse_array)
            self.neuron_list.append(committee_machine(input_size, hidden_size, n_synapse_array=n_synapse_array, weight_std_init=weight_std_init, add_bias=add_bias))

        
    def reinitiate(self):
        # Careful for bug: after reinitiate, the parameters for the optimizer may not be updated?
        for i in range(self.n_neuron):
            self.neuron_list[i].reinitiate()

    
    def recurrence(self,state_g):
        # synchronous update
        if state_g.dim() == 1:
            state_g = state_g.unsqueeze(0)
        state_g_new = torch.zeros_like(state_g)
        n_pattern = state_g.shape[0] 
        for i in range(self.n_neuron):
            input_lists = self.convert_pattern_to_input_list(state_g, self.dendrate[i], self.mask[i])
            # input_list = []
            # for j in range(self.hidden_size):
            #     input_list.append(state_g[(self.dendrate[i]==j) & self.mask[i]])
            for j in range(n_pattern):
                state_g_new[j,i], _, _ = self.neuron_list[i].forward_discrete(input_lists[j])
        return state_g_new.squeeze()

    
    def train(self, patterns, lr=0.01, training_max_iter=1000, method='ALA'):

        assert method in ['ALA', 'backprop']
        # Only apply to the case where kappa = 0
        success_network = []
        # for each neurons in the network, train the committee machine to store the corresponding pattern-label pairs
        for i in range(self.n_neuron):
            input_lists = self.convert_pattern_to_input_list(patterns, self.dendrate[i], self.mask[i])
            labels = patterns[:,i]
            # train the committee machine to store the pattern-label pair
            if method == 'ALA':
                success, _, _ = self.neuron_list[i].train_ALA(input_lists, labels, learning_rate=lr, max_epochs=training_max_iter)
            elif method == 'backprop':
                success, _, _ = self.neuron_list[i].train_backprop(input_lists, labels, learning_rate=lr, max_epochs=training_max_iter)
            success_network.append(success)
            # print('Neuron {} training complete'.format(i))
        success = torch.all(torch.tensor(success_network)).item()

        
        new_patterns = torch.zeros_like(patterns)
        for i in range(patterns.shape[0]):
            new_patterns[i] = self.recurrence(patterns[i])

        return success, new_patterns
    
    def convert_pattern_to_input_list(self, pattern, dendrate_idx, mask):
        """ pattern is a 2D tensor with the shape of (n_pattern, n_neuron),
        return a n_pattern list of n_hidden list of input vectors, each of which is a 1D tensor.
        dendrate_idx is a 1D tensor with the length of n_neuron, which shows the index of the dendrate that each item is reached to.
        mask is a 1D tensor with the length of n_neuron, which shows the mask of each neuron."""
        input_list = []
        n_pattern = pattern.shape[0]
        for i in range(n_pattern):
            input_list.append([])
            for j in range(self.hidden_size):
                input_list[i].append(pattern[i][(dendrate_idx==j) & mask])
        return input_list


# Example usage
if __name__ == "__main__":

    torch.random.manual_seed(18)
    np.random.seed(0)

    # # Train the committee machine model
    input_size = 500
    hidden_size = 10
    batch_size = 1000
    pattern_act_mean = 0.25
    training_max_iter = 10000
    n_synapse_array = torch.zeros((hidden_size,)).to(torch.int32)

    for i in range(hidden_size):
        n_synapse_array[i] = input_size // hidden_size
    n_synapse_array[-1] = n_synapse_array[-1] + input_size - torch.sum(n_synapse_array) # make sure the total number of synapses is equal to the input size

    # Create a random dataset
    labels = torch.sign(pattern_act_mean-torch.rand(batch_size))
    inputs = [[torch.sign(pattern_act_mean-torch.rand(n_synapse_array[i])) for i in range(hidden_size)] for _ in range(batch_size)]

    # Initialize the model
    model = committee_machine(input_size, hidden_size, n_synapse_array = n_synapse_array)
    # train the model
    train_log = model.train_backprop(inputs, labels, learning_rate=0.05, max_epochs=training_max_iter)


    outputs = torch.zeros(batch_size)
    for i in range(batch_size):
        outputs[i], _, _ = model.forward_discrete(inputs[i])
        input_tensor = model.pad_list_of_tensors(inputs[i]).unsqueeze(0)
        output_continue = model.forward_tensor_version(input_tensor)

    error_rate = torch.mean((outputs*labels<=0).float()).item()
    print('Error rate = {}'.format(error_rate))

    plt.figure()
    plt.plot(train_log[2], train_log[1])
    plt.xlabel('Epoch')
    plt.ylabel('Error rate')
    plt.show(block=True)
    plt.savefig('train_committee_machine.png')
    print('Training complete')

    # Train the random_connect_network_model:
    n_neuron = 100
    n_pattern = 40
    n_hidden = 1
    training_max_iter = 1000
    connection_prop = torch.ones((n_neuron,))*1
    learning_rate = 0.05
    pattern_act_mean = 0.25
    pattern_act_std = 0.0
    ratio_conn_mean = 0.5
    ratio_conn_std = 0.0
    change_in_degree = True
    change_out_degree = False

    # Train the random_connect_network_model:

    neuron_act_prop = (torch.rand((n_neuron,))-0.5)*(pattern_act_std*np.sqrt(12)) + pattern_act_mean
    if change_in_degree:
        connection_prop_in = (torch.rand((n_neuron,))-0.5)*(ratio_conn_std*np.sqrt(12)) + ratio_conn_mean
    else:
        connection_prop_in = torch.ones(n_neuron)*ratio_conn_mean
    if change_out_degree:
        connection_prop_out = (torch.rand((n_neuron,))-0.5)*(ratio_conn_std*np.sqrt(12)) + ratio_conn_mean
    else:
        connection_prop_out = torch.ones(n_neuron)*ratio_conn_mean

    mask_prob = (connection_prop_in[:, None] * connection_prop_out[None,:])/ratio_conn_mean
    mask_prob = torch.clip(mask_prob, 0, 1)
    mask = torch.bernoulli(mask_prob)
    mask.fill_diagonal_(0) # important: no self-connection

    network_model = random_connect_network(n_neuron=n_neuron, dt=0.01, hidden_size= n_hidden, weight_std_init=0.1, mask=mask, neuron_base_state=-1)
    # Create a random dataset
    patterns = utils.make_pattern(n_pattern, n_neuron, perc_active=neuron_act_prop)

    # Train the network
    intial_forward_patterns = network_model.recurrence(patterns)
    initial_error_rate = torch.mean((intial_forward_patterns*patterns<=0).float()).item()

    success, new_patterns = network_model.train(patterns, lr=learning_rate, training_max_iter=training_max_iter, method = 'backprop')

    error_rate = torch.mean((new_patterns*patterns<=0).float()).item()

    print('Training complete, initial error rate: {}, final error rate: {}'.format(initial_error_rate, error_rate))


