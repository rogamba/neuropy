from random import choice, gauss, random


class Gene(object):
    def __init__(self, config, key):
        self.config = config
        self.key = key

    def __str__(self):
        attrib = ['key'] + [a for a in self.__gene_attributes__]
        attrib = ['{0}={1}'.format(a, getattr(self, a)) for a in attrib]
        return '{0}({1})'.format(self.__class__.__name__, ", ".join(attrib))

    def clamp(self, name, value):
        ''' Limit the value in the specified range of min max
        '''
        min_value = self.config[ name+"_min_val" ]
        max_value = self.config[ name+"_max_val" ]
        return max(min(value, max_value), min_value)

    def init_attribute(self, name):
        ''' Init the value of the given attribute (weight, bias)
        '''
        # Float attributes
        #print("[gene] Init attributes")
        #print(name)
        if name in ['weight','bias','response']:
            mean = self.config[ name+"_init_mean" ]
            stdev = self.config[ name+"_init_stdev" ]
            return self.clamp(name, gauss(mean, stdev))
        # Boolean attributes
        if name in ['enabled']:
            default = self.config[ name+"_default" ]  
            return default if default != None else (random() < 0.5)
        # Activation and aggregation attribute
        if name in ['activation', 'aggregation']:
            default = self.config[ name+"_default" ]
            return default

    def init_attributes(self):
        ''' Loop attributes and set its initial value
        '''
        for attr in self.__gene_attributes__:
            setattr( self, attr, self.init_attribute(attr) )


    def crossover(self, couple):
        ''' Creates a new gene randomly ingeriting attributes form its parents
        '''
        # Instantiate the gene object
        child = self.__class__(self.config, self.key)
        for attr in self.__gene_attributes__:
            v1 = getattr(self, attr)
            v2 = getattr(self, attr)
            setattr( child, attr, v1 if random() > 0.5 else v2 )
        return child

    def copy(self):
        ''' Return a copied gene with its attributes
        '''
        new_gene = self.__class__(self.config, self.key)
        for attr in self.__gene_attributes__:
            setattr(new_gene, attr, getattr(self, attr))
        return new_gene



class NodeGene(Gene):

    __gene_attributes__ = [ 'bias','activation','aggregation','response' ]  

    def __init__(self, config, key=None, bias=None, activation=None, aggregation=None, response=None):
        #print("[nodegene] Init new nodegene")
        self.config = config
        self.key = key
        self.bias = bias
        self.activation = activation
        self.aggregation = aggregation
        self.response = response
        if self.bias == None or self.activation == None:
            self.init_attributes()

    def mutate(self):
        ''' Mutate bias, activation, aggregation, reponse
            TODO: add mutation for activation, aggregation & reponse
        '''
        # Bias
        r = random()
        # Replace bias
        replace_rate = float(self.config['bias_replace_rate'])
        mutate_rate = float(self.config['bias_mutate_rate'])
        if r < replace_rate:
            #print(">>> mutate replace float")
            self.bias = self.init_attribute('bias')
        # Mutate bias
        if r < replace_rate + mutate_rate:
            #print(">>> mutate modify float")
            self.bias = self.bias + gauss(0.0, self.config['bias_mutate_power'])
        self.bias = self.clamp('bias',self.bias)

    def distance(self, other):
        #print("[nodegene] [distance] ")
        #print("[nodegene] distance NODE1: "+str(self))
        #print("[nodegene] distance NODE1: "+str(other))
        
        d = abs(self.bias - other.bias) + abs(self.response - other.response)
        #print("[nodegene] Checking distance: "+str(d))
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        return d * self.config['compatibility_weight_coefficient']



class EdgeGene(Gene):

    __gene_attributes__ = [ 'weight','enabled' ]  

    def __init__(self, config, key=None, weight=None, enabled=None):
        #print("[edgegene] Init new edgegene")
        self.key=key
        self.config = config
        self.weight=weight
        self.enabled=enabled
        if self.weight == None or self.enabled == None:
            self.init_attributes()

    def __lt__(self, other):
        return self.key < other.key

    def mutate(self):
        ''' Mutate edge gene attributes:  weight, enabled
        '''
        # Weight
        r = random()

        replace_rate = float(self.config['weight_replace_rate'])
        mutate_rate = float(self.config['weight_mutate_rate'])
        # Replace weight
        if r < replace_rate:
            #print(">>> mutate replace float")
            self.weight = self.init_attribute('weight')
        # Mutate weight
        if r < replace_rate + mutate_rate:
            #print(">>> mutate modify float")
            self.weight = self.weight + gauss(0.0, self.config['weight_mutate_power'])
        self.weight = self.clamp('weight',self.weight)   
        # Mutate enabled
        r = random()
        if r < self.config['enabled_mutate_rate']:
            #print(">>> mutate bool")
            self.enabled = (random() < 0.5)
        

    def distance(self, other):
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        return d * self.config['compatibility_weight_coefficient']
