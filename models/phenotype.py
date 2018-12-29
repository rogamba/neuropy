import numpy as np
from models.neural_net import NeuralNet
import config

''' Phenotype is the "real" or functional representation
    of a genotype. In this case, the phenotype refers to
    a neural network itself.
'''

class Phenotype():

    def __init__(self, config, genome):
        ''' Given a genotype build its phenotype
        '''
        #print("[phenotype] Init phenotype object")
        #print("[phenotype] Genome: "+str(genome))
        self.config = config
        # Set the genome
        self.genome = genome
        self.inputs = self.genome.input_keys
        self.outputs = self.genome.output_keys
        self.edges = self.genome.edges_list 
        # Properties to build the netowork
        self.layers = None 
        self.weights = None
        self.bias = None
        self.activation = None
        self.format()


    def output_path_nodes(self):
        ''' Obtiene desde la salida, el path de los nodos necesarios para el output
            Get required nodes to collect the final output
            given the input and output keys
            return a list of required nodes in layer format
        ''' 
        required = set(self.outputs)
        s = set(self.outputs)
        while 1:
            # Find nodes not in s whose output is consumed by a node in s.
            # Find the output nodes that lead to the global output
            t = set(a for (a, b) in self.edges if b in s and a not in s)
            if not t:
                break
            # NO inputs in the set
            layer_nodes = set(x for x in t if x not in self.inputs)
            if not layer_nodes:
                break
            required = required.union(layer_nodes)
            s = s.union(t)
        return required



    def format(self):
        ''' Get layers of a genotype from its link and node genes:
            Teniendo las keys de las neuronas input y output podemos construir las capas de la red 
            Layers:     [ {<key11>,<key12>}, {<key21>,<key22>} ]
        '''
        required = self.output_path_nodes()
        # Get the layers
        layers = []
        s = set(self.inputs)
        while 1:
            # Find candidate nodes c for the next layer.  These nodes should connect
            # a node in s to a node not in s.
            c = set(b for (a, b) in self.edges if a in s and b not in s)
            # Keep only the used nodes whose entire input set is contained in s.
            t = set()
            for n in c:
                if n in required and all(a in s for (a, b) in self.edges if b == n):
                    t.add(n)
            if not t:
                break
            layers.append(t)
            s = s.union(t)
        self.layers = layers
        # Get weights by layers



    def create(self):
        ''' Receives a genome and returns its phenotype (a FeedForwardNetwork).
            The format required to create the net is:
            Weights = [ [HiddenMatrix], [OutputsMatrix] ] 
        '''
        # Gather expressed connections.
        edges = [eg.key for eg in self.genome.edges.values() if eg.enabled]

        # Build layers
        self.format()
        
        node_evals = []
        # Building the layers and neurons
        for layer in self.layers:
            # Iterate the nodes for each layer
            for node in layer:
                inputs = []
                node_expr = []

                for eg in self.genome.edges.values():
                    inode, onode = eg.key
                    if onode == node and eg.enabled:
                        inputs.append((inode, eg.weight))
                        node_expr.append("v[{}] * {:.7e}".format(inode, eg.weight))

                ng = self.genome.nodes[node]
                aggregation_function = ng.aggregation
                activation_function = ng.activation
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        return NeuralNet(self.inputs, self.outputs, node_evals)

    