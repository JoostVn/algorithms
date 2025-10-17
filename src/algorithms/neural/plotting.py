import numpy
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from .nn import Layer, NeuralNetwork

class NetworkPlotter:
    
    def __init__(self, network, scale=1):
        self.n = network
        self.node_coords, self.edge_coords = self.get_coordinates(scale)
        self.node_biases, self.edge_weights = self.get_weights_and_biases()
        self.color_red_green = np.vectorize(self._color_red_green)

    def _color_red_green(self, value):
        """
        Chooses a red or green color based on a given value.
        """
        return to_rgb('red' if value < 0 else 'green')
        
    def rgba_picker(self, values, alpha=True):
        """
        Returns the RGBA colors for an array of values. If alpha=False, only
        returns RGB colors.
        """
        rgb_cols = self.color_red_green(values)
        if alpha:
            return np.vstack((rgb_cols, abs(values))).T
        else:
            return rgb_cols
       
    def get_coordinates(self, scale):
        """
        Computes network node and connection coordinates for plotting.
        """
        # Dimensions of each layer
        layer_dims = [self.n.input_dim] + [l.size for l in self.n.layers]
        
        # Layer node coordinates including input layer
        layer_nodes = []
        for i, size in enumerate(layer_dims):
            x = np.full(size, i*scale)
            y = (np.arange(0, size, 1) - (size-1)/2) * scale
            layer_nodes.append(np.vstack((x,y)))
        
        # Node pair connections
        layer_edges = []
        for cur, prev in zip(layer_nodes[1:], layer_nodes[:-1]):
            layer_edges += [
                np.vstack((i, j)).T for i in cur.T for j in prev.T]
            
        # Stack and return all nodes and connections
        nodes = np.hstack(layer_nodes)
        edges = np.array(layer_edges)
        return nodes, edges
     
    def get_weights_and_biases(self):
        """
        Fetches flat vectors of weights and biases from the net.
        """
        weights = np.concatenate([l.weights.flatten() for l in self.n.layers])
        biases = np.concatenate([l.biases for l in self.n.layers])
        return biases, weights

    def get_text(self, coordinates, values, offset):
        """
        Returns text coordinates and strings given a list of node coordinates,
        node values and a text offset.
        """
        text_pos = (coordinates.T - np.array(offset)).T
        text_str = values.round(2).astype(str)
        return text_pos, text_str

    def pyplot_structure(
            self, ax, node_size=800, font_size=12, font_offset=(0,0)):
        """
        Matplotlib plot of network structure.
        """
        # Input + hidden + output layer nodes
        input_layer = self.node_coords.T[:self.n.input_dim].T
        ax.scatter(*input_layer, s=node_size, c='grey', zorder=20)
        layers = self.node_coords.T[self.n.input_dim:].T
        cols = self.rgba_picker(self.node_biases)
        ax.scatter(*layers, s=node_size, c='white', zorder=10)
        ax.scatter(*layers, s=node_size, c=cols, edgecolors='grey', linewidths=0.5, zorder=20)
        
        # Layer edges
        cols = self.rgba_picker(self.edge_weights)
        for connection, col in zip(self.edge_coords, cols):
            ax.plot(*connection, color=col, zorder=5, linewidth=0.5)
        
        # Node bias text
        text_pos, text_str = self.get_text(layers, self.node_biases, font_offset)
        annotate = np.vectorize(ax.text)
        annotate(
            *text_pos, text_str, fontsize=font_size, zorder=100, ha="center",
            va="center")
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
    def pyplot_forward_pass(self, ax, node_values, edge_values, node_size=800, 
                            font_size=12, font_offset=[0.08,0.35]):
        """
        Matplotlib plot of edge values and node activations from a forward pass.
        """
        # Flattening value arrays
        edgevals = np.concatenate([e.flatten() for e in edge_values])
        nodevals = np.concatenate(node_values)
        
        # Node layers
        cols = self.rgba_picker(nodevals)
        ax.scatter(*self.node_coords, s=node_size, c='white', zorder=10)
        ax.scatter(*self.node_coords, s=node_size, c=cols, zorder=20)
        
        # Layer edges
        cols = self.rgba_picker(edgevals)
        for connection, col in zip(self.edge_coords, cols):
            ax.plot(*connection, color=col, zorder=5)
        
        # Node activation text
        text_pos, text_str = self.get_text(self.node_coords, nodevals, font_offset)
        annotate = np.vectorize(ax.text)
        annotate(*text_pos, text_str, fontsize=font_size)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')