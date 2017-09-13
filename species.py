from indexer import Indexer

class Species(object):

    def __init__(self, config, key, generation):
        ''' Representation of a single Species object,
            Each species object have: genomes (list of members)
        '''
        #print("[species] Init species: key %s | gen %s " % (key,generation))
        self.config = config
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.generation = None
        self.representative = None
        self.genomes = None
        self.fitness_history = []
        self.fitness = None
        self.adjusted_fitness = None

        
    def __str__(self):
        s = "Species:" +str(self.key)
        s += "\n\tMembers: "+str(len(self.genomes))
        s += "\n\tAdjusted Fitness: "+str(self.adjusted_fitness)
        s += "\n\tGeneration: "+str(self.generation)
        s += "\n\tFitness: "+str(self.fitness)
        return s


    def update(self, representative, genomes):
        self.representative = representative
        self.genomes = genomes

    def get_fitnesses(self):
        ''' Get list of fitnesses from the members of the species
        '''
        return [g.fitness for g in self.genomes.values()]





class DistanceCache(object):

    def __init__(self, config):
        #print("[distance] Init distance cache")
        self.distances = {}
        self.config = config
        self.hits = 0
        self.misses = 0

    def __call__(self, genome0, genome1):
        #print("[distance cache]")
        #print(genome0)
        #print(genome1)
        #print("[distance] getting distance between: "+str(genome0.key)+ " and "+str(genome1.key))
        g0 = genome0.key
        g1 = genome1.key
        d = self.distances.get((g0, g1))
        if d is None:
            # Distance is not already computed.
            d = genome0.distance(genome1)
            self.distances[g0, g1] = d
            self.distances[g1, g0] = d
            self.misses += 1
        else:
            self.hits += 1

        return d