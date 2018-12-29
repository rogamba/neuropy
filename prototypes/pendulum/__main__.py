from models.evolution import Evolution
from models.phenotype import Phenotype
import config
from models import visualize
import numpy as np
import os
from .model import PendulumCart

''' Steps to use the library:
    1. Set the configuration parameters
    2. Import the configuration variables
    3. Set the fitness function you would like to eval your genomes with
    4. Pass the config dict to the evolution instance
    5. RUn the evolution process passing the fitness function
        
'''

REL_PATH = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = REL_PATH+'/config.json'


np.set_printoptions(suppress=True)
v = np.array([0,0,0,0,0,0])
cart = PendulumCart()


# Define fitness evaluation function for each genome
def eval_genome(genome):
    ''' Receives a list of genome objects and sets 
        the fitness to every genome
    '''
    global cart
    global v
    global w
    # Create phenotype from the genome
    phenotype = Phenotype(config.params, genome)
    net = phenotype.create()

    # How to calculate the fitness of the genome
    # Evaluate the genome for the maximum time


    # Fitness = avg time of all the episodes
    # How many episodes for the same net?
    
    times = []
    #print("-------------> new genome")
    for e in range(3):
        step = 0
        done = False
        s = cart.reset()
        a = 0
        for n in range(1,1000):
            # Render
            cart.render()
            s_, r, done, info = cart.step(a)

            # How many steps did the cart hold?
            step += 1
            output = net.activate(s_)
            a_ = output[0]
            print("Net Output: ")
            print(a_)

            # Update state and action
            s = s_
            a = a_

            if done or n >= 1000:
                print("Episode finished after %s steps" % (step,))
                times.append(step)
                break

    # Get the genome fitness (avg steps) * 10 = seconds
    fitness = np.mean(times) / 50
    print("Genome fitness: ",fitness)
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



    #for xi, xo in zip(xor_inputs, xor_outputs):
    #    output = winner_net.activate(xi)
    #    print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))




    # Print the net
    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, view=True, node_names=node_names,filename=REL_PATH+"plots/winner.gv")



if __name__ == '__main__':
    run()