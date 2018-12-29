import numpy as np
import sys
import random
from models.activations import get_activation

''' Developer:  Rodrigo Gamba
    Neural net class, for evolving
'''


class NeuralNet(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise Exception("Expected {0} inputs, got {1}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            # Functions
            activ = get_activation(act_func)
            aggreg = eval(agg_func)
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            s = aggreg(node_inputs)
            self.values[node] = activ(bias + response * s)

        return [self.values[i] for i in self.output_nodes]



class NeuralNetwork:

    layers=None                 # Number of layers
    W=[]                        # List of weights
    b=[]
    f=[]
    a=[]
    n=[]
    s=[]
    p=[]
    t=[]


    def __init__(self,layers=2,W=W,b=b,f=f):
        ''' Init the network
        '''
        self.layers = layers        # int of number of layers
        self.W = W                  # list of weight matrices by layer
        self.b = b                  # list of bias vectors by layer
        self.f = f                  # list of functions by layer
        self.F = f                  # list of matrices of function derivatives valuated
        pass   


    def activate(self):
        ''' Feedforward pass for the neural network
        '''
        self.a = []
        self.n = []
        # Iterate layers
        for i in range(0,self.layers):
            # a^m-1 = p if i = 0
            act = self.pk if len(self.a) <= 0 else self.a[i-1]
            # Get input vector of layer
            self.n.append( np.dot(self.W[i],act) + self.b[i] )
            # Apply activation to input 
            self.a.append( self.f[i](self.n[-1]) )
