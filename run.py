from .evolution import Evolution
from .phenotype import Phenotype
from . import config
from . import visualize

''' Steps to use the library:
    1. Set the configuration parameters
    2. Import the configuration variables
    3. Set the fitness function you would like to eval your genomes with
    4. Pass the config dict to the evolution instance
    5. RUn the evolution process passing the fitness function
'''

# Given the name of a simulation, include it and run it


CONFIG_FILE = config.PATH+'config.json'
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

# Define fitness evaluation function for each genome
def eval_genome(genome):
    ''' Receives a list of genome objects and sets 
        the fitness to every genome
    '''
    # Create phenotype from the genome
    phenotype = Phenotype(config.params, genome)
    net = phenotype.create()
    
    # Set fitness
    fitness=4
    # Calculate fitness with te behaviour of the net 
    for xi, xo in zip(xor_inputs, xor_outputs):
        # Test network with the input
        output = net.activate(xi)
        # Error squared
        fitness -= (output[0] - xo[0]) ** 2  
    #print("[run] genome fitness: %s " % (fitness,))  
    return fitness



def run():
    print("[run] Starting main")
    # Get configuration parameters
    config.build(CONFIG_FILE)

    # Init the evolution object
    evolution = Evolution(config.params)

    # Run evolution
    winner = evolution.run(fitness_function=eval_genome,generations=config.params['generations'])
    
    # Show performance of winner vs actual value
    #evaluate_winner

    print('\nOutput:')
    ph = Phenotype(config.params, winner)
    winner_net = ph.create()
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    # Print the net
    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config.params, winner, True, node_names=node_names, filename="plots/winner.gv")



if __name__ == '__main__':
    run()