from gene import NodeGene, EdgeGene
from phenotype import Phenotype
import random
from pprint import pprint


class Genome(object):

    def __init__(self, config, key, init=True, nodes=[], edges=[]):
        ''' Create a randome genome by default
            self.nodes = { <key> : <object NodeObject>, ... }  
            self.connections = { (<node_in>,<node_out>) : <object NodeObject>, ... }  
        '''
        #print("[genome] Init new genome")
        self.config = config
        self.key = key
            
        self.edges={}
        self.nodes={}

        # Init the nodes and edges genes of the genome
        if nodes and edges:
            for node in nodes:
                self.nodes[node.key] = node
            for edge in edges:
                self.edges[edge.key] = edge

        self.fitness = None
        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.config['num_inputs'])]
        self.output_keys = [i for i in range(self.config['num_outputs'])]
        # Activations -> the nodes init_attributes method takes care of this
        self.activation = self.config['activation_default']

        # Create new genome
        if init and not (nodes and edges):
            self.new()


    def __str__(self):
        s="Key: "+str(self.key) +"\n"   
        s += "Nodes:"
        for k, ng in self.nodes.items():
            s += "\n\t{0} {1!s}".format(k, ng)
        s += "\nConnections:"
        edges = list(self.edges.values())
        edges.sort()
        for c in edges:
            s += "\n\t" + str(c)
        s+="\nFitness\n\t" + str(self.fitness)
        
        return s


    def new(self):
        ''' Create a new genome from the given configurations
        '''
        # Create node genes for the output pins
        for node_key in self.output_keys:
            self.nodes[node_key] = NodeGene(self.config, node_key)

        # Add hidden nodes if requested
        if self.config['num_hidden'] > 0:
            for i in range(self.config['num_hidden']):
                node_key = self.new_node_key()
                assert node_key not in self.nodes
                node = NodeGene(self.config, node_key)
                self.nodes[node_key] = node

        # Add connections on initial connectivity type
        self.connect()


    def crossover(self, genome1, genome2):
        ''' Reconfigures the genome given two parent genomes
        '''
        #print("[genome] crossover")
        # Check which parent is fittest
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1
        # Inherit connection genes
        for key, eg1 in parent1.edges.items():
            eg2 = parent2.edges.get(key)
            if eg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.edges[key] = eg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.edges[key] = eg1.crossover(eg2)
                
        # Inherit node genes
        nodes1 = parent1.nodes
        nodes2 = parent2.nodes
        #print("------------------- Checking nodes")
        #pprint(nodes1)
        #pprint(nodes2)
        for key, ng1 in nodes1.items():
            ng2 = nodes2.get(key)
            #print(self.nodes)
            #pprint(key)
            #pprint(self.nodes)

            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)



    def full_connections(self):
        ''' Compute connections for a fully connected feed-forward network
        '''
        # Compute full connections of the network
        hidden = [i for i in self.nodes.keys() if i not in self.output_keys]
        output = [i for i in self.nodes.keys() if i in self.output_keys]
        connections = []
        if hidden:
            for input_id in self.input_keys:
                for h in hidden:
                    connections.append((input_id, h))
            for h in hidden:
                for output_id in output:
                    connections.append((h, output_id))
        else:
            for input_id in self.input_keys:
                for output_id in output:
                    connections.append((input_id, output_id))

        # For recurrent genomes, include node self-connections.
        if self.config['net_type'] == 'recurrent':
            for i in self.nodes.keys():
                connections.append((i, i))
        return connections


    def connect(self):
        ''' Fully connect node genes of the genome, add the required
            connection genes to fully connect
        '''
        for input_key, output_key in self.full_connections():
            edge = EdgeGene(self.config,(input_key,output_key))
            self.edges[edge.key] = edge




    def new_node_key(self):
        ''' Getting a new innovation number (key) 
            given the number of nodes in the genome
        '''
        new_id = 0
        while new_id in self.nodes:
            new_id += 1
        return new_id

    
    def mutate(self):
        ''' Mutate the genome evaluating random vs the config parameters
            for nodes, connections
            Mutate rates in config file
        '''
        #print("[genome] mutating")
        m=0
        if random.random() < self.config['node_add_prob']:
            m+=1
            #print("[genome] mutate add node")
            self.mutate_add_node()
        if random.random() < self.config['node_delete_prob']:
            m+=1
            #print("[genome] mutate del node")
            self.mutate_delete_node()
        if random.random() < self.config['edge_add_prob']:
            m+=1
            #print("[genome] mutate add edge")
            self.mutate_add_edge()
        if random.random() < self.config['edge_delete_prob']:
            m+=1
            #print("[genome] mutate del edge")
            self.mutate_delete_edge()
        #print("Mutations: "+str(m))
        # Check mutation of attributes in the edge nodes (enabled, weight)
        for eg in self.edges.values():
            eg.mutate()
        # Mutate node genes (bias )
        for ng in self.nodes.values():
            ng.mutate()
        


    def mutate_add_node(self):
        ''' Adding node equals to: splitting edge, adding node between two nodes
            setting one weight of the connection to 1 and the other as the old weight.
            The new node and connections hace roughly the same behaviour as the original 
            connection.
        '''
        if not self.edges:
            return None, None
        # Choose a random connection to split
        edge_to_split = random.choice(list(self.edges.values()))
        node_key = self.new_node_key()
        ng = NodeGene(self.config, node_key)
        # Set the node object in the nodes dict
        self.nodes[node_key] = ng

        edge_to_split.enabled = False

        i, o = edge_to_split.key
        self.add_edge(i, node_key, 1.0, True)
        self.add_edge(node_key, o, edge_to_split.weight, True)


    def mutate_delete_node(self):
        ''' Delete node from genome
            Do nothing if there are no non-output nodes
        '''
        # Do nothing if there are no non-output nodes.
        available_nodes = [(k, v) for k, v in self.nodes.items() if k not in self.output_keys]
        if not available_nodes:
            return -1

        del_key, del_node = random.choice(available_nodes)
        edges_to_delete = set()
        for k, v in self.edges.items():
            if del_key in v.key:
                edges_to_delete.add(v.key)

        for key in edges_to_delete:
            del self.edges[key]

        del self.nodes[del_key]

        return del_key


    def mutate_add_edge(self):
        ''' Attempt to add a new connection, the only restriction being that the output
            node cannot be one of the network input pins.
        '''
        possible_outputs = list(self.nodes.keys())
        out_node = random.choice(possible_outputs)

        possible_inputs = possible_outputs + self.input_keys
        #print("[mutate] possible inputs: "+ str(possible_inputs))
        in_node = random.choice(possible_inputs)

        # Don't duplicate connections. The proposed connection already exists
        key = (in_node, out_node)
        if key in self.edges:
            return

        # For feed-forward networks, avoid creating cycles.
        if self.config['net_type'] == 'feed_forward' and self.creates_cycle(list(self.edges.keys()), key):
            return

        cg = EdgeGene(self.config,(in_node, out_node))
        self.edges[cg.key] = cg


    def mutate_delete_edge(self):
        if self.edges:
            key = random.choice(list(self.edges.keys()))
            del self.edges[key]


    def add_edge(self, input_key, output_key, weight, enabled):
        ''' Add edge (connection) to the edges dict with the 
            given parameters
        '''
        key = (input_key, output_key)
        edge = EdgeGene(self.config,key)
        edge.weight = weight
        edge.enabled = enabled
        self.edges[key] = edge


    def distance(self, other):
        ''' Compute the distance from two genomes
        '''
        # Compute node gene distance component
        node_distance = 0.0
        if self.nodes or other.nodes:
            # Check disjoint nodes from 2 to 1
            disjoint_nodes = 0
            for k2 in other.nodes.keys():
                if k2 not in self.nodes:
                    disjoint_nodes += 1
            # Check disjoint from 1 to 2
            for k1, n1 in self.nodes.items():
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2)
            
            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance + self.config['compatibility_disjoint_coefficient'] * disjoint_nodes) / max_nodes

        # Compute edge gene differences.
        edge_distance = 0.0
        if self.edges or other.edges:
            disjoint_edges = 0
            for k2 in other.edges.keys():
                if k2 not in self.edges:
                    disjoint_edges += 1

            for k1, e1 in self.edges.items():
                e2 = other.edges.get(k1)
                if e2 is None:
                    disjoint_edges += 1
                else:
                    # Homologous genes compute their own distance value.
                    edge_distance += e1.distance(e2)

            max_edge = max(len(self.edges), len(other.edges))
            edge_sum = edge_distance + self.config['compatibility_disjoint_coefficient'] * disjoint_edges
            edge_distance = edge_sum / max_edge
        # Sum distances
        distance = node_distance + edge_distance
        return distance


    def creates_cycle(self, edges, test):
        """ Returns true if the addition of the "test" connection would create a cycle,
            assuming that no cycle already exists in the graph represented by "connections".
        """
        i, o = test
        if i == o:
            return True

        visited = {o}
        while True:
            num_added = 0
            for a, b in edges:
                if a in visited and b not in visited:
                    if b == i:
                        return True

                    visited.add(b)
                    num_added += 1

            if num_added == 0:
                return False


    # [Pending]
    def pair(self, genome):
        ''' Breed a child given two genomes
            1. Considerar únicamente los genes de las conexiones (link genes)
            2. Se alinean los genes con el mismo innovation number
            3. De los genes alineados, los que se heredan al hijo se hacen aleatoriamente
            4. Los disjoint y excess siempre son heredados del gen con mayor fitness
        '''
        pass
        


    @property
    def edges_list(self):
        return [eg.key for eg in self.edges.values() if eg.enabled]

    @property
    def nodes_list(self):
        return [ng.key for ng in self.nodes.values()]