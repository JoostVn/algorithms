import numpy
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb


class Layer:
    
    def __init__(self, size, input_dim, activation_func=np.tanh):
        """
        On neural network layer. 
        
        Contains a column of nodes and weights for connections to the previous 
        layer.
        """
        self.size = size
        self.input_dim = input_dim
        self.biases = np.zeros(size)
        self.weights = np.zeros((size, input_dim))
        self.activation_func = activation_func

    def __repr__(self):
        return f'\nBiases\n{self.biases}\n\nWeights\n{self.weights}\n'

    @property
    def genome_size(self):
        return len(self.get_genome())

    def random_init(self):
        """
        Set all weights and biases to random uniform values.
        """
        domain = (-1, 1)
        self.biases = np.random.uniform(*domain, self.biases.shape)
        self.weights = np.random.uniform(*domain, self.weights.shape)
        
    def fire(self, input_values):
        """
        Pass input_values trough layer and return node activations.
        """
        return self.activation_func(self.weights.dot(input_values) + self.biases)
    
    def fire_only_weights(self, input_values):
        """
        Debug mode: Only multiply input_values with edge weights.
        """
        return input_values * self.weights
        
    def fire_only_biases(self, input_edge_product):
        """
        Debug mode: Only add biases to input_edge_product.
        """
        return input_edge_product + self.biases
        
    def fire_only_activation(self, weight_bias_productsum):
        """
        Debug mode: Only apply activation function to weight_bias_productsum.
        """
        return self.activation_func(weight_bias_productsum)
    
    def get_genome(self):
        """
        Return genome as a flat array of (biases, weights).
        """
        return np.concatenate((self.biases, self.weights.flatten()))
        
    def set_genome(self, genome):
        """
        Set genome from a flat array of (biases, weights).
        """
        self.biases = genome[:self.size]
        self.weights = genome[self.size:].reshape(self.weights.shape)



class NeuralNetwork:
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.layers = []

    @property
    def genome_size(self):
        """
        Return the total lenght of the neural network genome.
        """
        return sum([layer.genome_size for layer in self.layers])

    def add_layer(self, layer):
        """
        Add one layer to the end of the network. Also updates weights/biases.
        """
        self.layers.append(layer)
        
    def random_init(self):
        """
        Set all layer weights and biases to random uniform values.
        """
        for layer in self.layers:
            layer.random_init()
            
    def forward_pass(self, input_values):
        """
        Forward pass of the neural network based on an array of input values.
        """
        for layer in self.layers:
            output_values = layer.fire(input_values)
            input_values = output_values
        selection = np.argmax(output_values)
        return output_values, selection

    def forward_pass_debug(self, input_values):
        """
        Forward pass of the neural network based on an array of input values.
        Returns not only the network output, but the value of edges and 
        nodes for each intermediate step. Edge values are defined as 
        (edge weigth * prev node output) and node values are defined as
        the output of their activation function.
        """
        node_values = [input_values]
        edge_values = []
        for layer in self.layers:
            input_edge_product = layer.fire_only_weights(input_values)
            product_sum = input_edge_product.sum(axis=1)
            biases_sum = layer.fire_only_biases(product_sum)
            node_activations = layer.fire_only_activation(biases_sum)
            edge_values.append(input_edge_product)
            node_values.append(node_activations)
            input_values = node_activations
        output_values = node_activations
        selection = np.argmax(node_activations)
        return output_values, selection, node_values, edge_values

    def get_genome(self):
        """
        Retreives all weight and biases in the network. Genome stucture:
        (l1_biases, l1_weights, ..., ln_biases, ln_weights)
        """
        return np.concatenate([layer.get_genome() for layer in self.layers])

    def set_genome(self, genome):
        """
        sets all weight and biases in the network. Genome structure:
        (l1_biases, l1_weights, ..., ln_biases, ln_weights)
        """
        for layer in self.layers:
            layer.set_genome(genome[:layer.genome_size])
            genome = genome[layer.genome_size:]