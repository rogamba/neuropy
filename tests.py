import unittest
from evolution import Evolution
from population import Population
from genome import Genome
from gene import NodeGene, EdgeGene
from species import Species
from neural_net import NeuralNet
from phenotype import Phenotype
from models.activations import *
from run import eval_genome as fitness_function
import config


params = config.build(config.PATH+'config.json')
sigmoid = sigmoid_activation


class PriceTestClass(unittest.TestCase):


    def test_create_new_nodegene(self):
        ''' Testing create nodeGene
        '''
        node = NodeGene(params, key=1, bias=0.1, activation="sigmoid", aggregation="sum", response=1.0)
        self.assertEqual(1,1)

    def test_create_new_edgegene(self):
        ''' Testing create a new edgeGene
        '''
        edge = EdgeGene(params, key=(-1, 0), weight=0.40031, enabled=True)
        self.assertEqual(1,1)


    def test_create_new_genome(self):
        ''' Create a new genome given some node and edge genes
        '''
        genome = Genome(
            params, 1,
            nodes = [
                NodeGene(params, key=0, bias=-0.1736240132898755, activation="sigmoid", aggregation="sum", response=1.0),
                NodeGene(params, key=1, bias=-1.089662414942449, activation="sigmoid", aggregation="sum", response=1.0)
            ],
            edges = [
                EdgeGene(params, key=(-2, 1), weight=-2.0917836717324634, enabled=True),
                EdgeGene(params, key=(-1, 0), weight=0.14527420648479528, enabled=True) 
            ]
        )
        #print(genome)
        self.assertEqual(1,1)

    def test_setting_genome_fitness(self):
        ''' Create a new genome given some node and edge genes
        '''
        genome = Genome(
            params, 1,
            nodes = [
                NodeGene(params, key=0, bias=-0.1736240132898755, activation="sigmoid", aggregation="sum", response=1.0),
                NodeGene(params, key=1, bias=-1.089662414942449, activation="sigmoid", aggregation="sum", response=1.0)
            ],
            edges = [
                EdgeGene(params, key=(-2, 1), weight=-2.0917836717324634, enabled=True),
                EdgeGene(params, key=(-1, 0), weight=0.14527420648479528, enabled=True) 
            ]
        )
        genome.fitness = fitness_function(genome)
        #print(genome)
        self.assertEqual(1,1)


    def test_create_new_child(self):
        parent1 = Genome(
            params, 1,
            nodes = [
                NodeGene(params, key=0, bias=0.011914079115631561, activation="sigmoid", aggregation="sum", response=1.0)
            ],
            edges = [
                EdgeGene(params, key=(-2, 0), weight=0.20650233514641234, enabled=True),
                EdgeGene(params, key=(-1, 0), weight=-0.7724663301453796, enabled=False)
            ]
        )
        parent1.fitness = fitness_function(parent1)
        print(parent1)
        parent2 = Genome(
            params, 2,
            nodes = [
                NodeGene(params, key=0, bias=0.24350343348399164, activation="sigmoid", aggregation="sum", response=1.0),
                NodeGene(params, key=1, bias=-1.5006387138462196, activation="sigmoid", aggregation="sum", response=1.0)
            ],
            edges = [
                EdgeGene(params, key=(-2, 0), weight=0.2762525844724594, enabled=False),
                EdgeGene(params, key=(-2, 1), weight=0.46719911418408594, enabled=True),
                EdgeGene(params, key=(1, 0), weight=-0.23587815688724798, enabled=True)
            ]
        )
        parent2.fitness = fitness_function(parent2)
        print(parent2)
        child = Genome(params, 3, init=False)
        child.crossover(parent1, parent2)
        child.fitness = fitness_function(child)
        print(child)
        self.assertEqual(1,1)

    def test_crossover_genomes(self):
        pass

    def test_mutate_gene(self):
        pass

    def test_mutate_genome(self):
        pass


    def test_create_phenotype(self):
        pass



# runs the unit tests in the module
if __name__ == '__main__':
    unittest.main()
