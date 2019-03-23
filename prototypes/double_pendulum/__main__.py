from models.evolution import Evolution
from models.phenotype import Phenotype
from .model import DoublePendulumCart
import models.visualize as visualize
import config
import numpy as np
import os
import sys
import json
import sys
import datetime

''' Steps to use the library:
    1. Set the configuration parameters
    2. Import the configuration variables
    3. Set the fitness function you would like to eval your genomes with
    4. Pass the config dict to the evolution instance
    5. RUn the evolution process passing the fitness function

    x = state.item(0)
    theta = state.item(1)
    phi = state.item(2)
    x_dot = state.item(3)
    theta_dot = state.item(4)
    phi_dot = state.item(5)
        
'''

REL_PATH = os.path.dirname(os.path.relpath(__file__))
CONFIG_FILE = REL_PATH+'/config.json'

np.set_printoptions(suppress=True)
gamma = 0.99
beta = 0.00001
alpha = 0.000001
sigma = 0.001
w = np.array([0, 0, 0, 0, 0, 0, 0, 0])
delta_w = np.array([0,0,0,0,0,0,0,0])
v = np.array([0,0,0,0,0,0,0,0])
delta_v = np.array([0,0,0,0,0,0,0,0])
cart = DoublePendulumCart()


def sign(x):
    if (x > 0):
        output = 1
    elif x < 0:
        output = -1
    else:
        output = 0
    return output

def normalize_angle(angle):
    """
    3*pi gives -pi, 4*pi gives 0 etc, etc. (returns the negative difference
    from the closest multiple of 2*pi)
    """
    normalized_angle = abs(angle)
    normalized_angle = normalized_angle % (2*np.pi)
    if normalized_angle > np.pi:
        normalized_angle = normalized_angle - 2*np.pi
    normalized_angle = abs(normalized_angle)
    return normalized_angle

def phi_critic(state, action):
    x = state.item(0)
    theta = state.item(1)
    phi = state.item(2)
    x_dot = state.item(3)
    theta_dot = state.item(4)
    phi_dot = state.item(5)
    phi_critic = np.array([normalize_angle(theta),
            normalize_angle(phi), theta*theta_dot, phi*phi_dot,
            action*theta, action*phi,
            action*theta_dot, action*phi_dot])
    return phi_critic
    
def q_hat(state, action, w):
    X = phi_critic(state, action)
    output = np.dot(X,w)
    return output
    
def phi_actor(state):
    #print(state)
    x = state.item(0)
    theta = state.item(1)
    phi = state.item(2)
    x_dot = state.item(3)
    theta_dot = state.item(4)
    phi_dot = state.item(5)
    ind1 = 0
    ind2 = 0
    if theta > 0 and phi < 0:
        ind1 = 1
    if theta < 0 and phi > 0:
        ind2 = 1
    phi_actor = np.array([theta, phi, theta_dot, phi_dot, sign(theta), sign(phi), ind1, ind2])
    return phi_actor
    
def mu(state, v):
    X = phi_actor(state)
    output = np.dot(X,v)
    return output
    


# Define fitness evaluation function for each genome
def eval_genome(genome, test=False):
    ''' Receives a list of genome objects and sets 
        the fitness to every genome
    '''
    global cart
    global v
    global w

    # Create phenotype from the genome
    phenotype = Phenotype(config.params, genome)
    net = phenotype.create()
    #cart = DoublePendulumCart()

    # How to calculate the fitness of the genome
    # Evaluate the genome for the maximum time


    # Fitness = avg time of all the episodes
    # How many episodes for the same net?
    
    times = []
    for e in range(config.params['evals_per_genome']):
        step = 0
        done = False
        s = cart.reset()
        a = 0
        for n in range(1,5000):

            # Render
            if test or ('animation' not in config.params or config.params['animation']):
                cart.render()

            # Cart state    
            s_, r, done, info = cart.step(a)

            if 'logging' not in config.params or config.params['logging']:
                print("Cart state: ")
                print(s_)

            # How many steps did the cart hold?
            step += 1
            output = net.activate(s_)
            a_ = output[0]

            # update v (actor): ??
            #delta_v = alpha*(((a - mu(s, v))*phi_actor(s)))*q_hat(s, a, w)
            #v = np.add(v, delta_v)
 
            # update w (critic): ??
            #delta_w = (beta*(r + gamma*q_hat(s_, a_, w) - q_hat(s, a, w)))*phi_critic(s, a)
            #w = np.add(w, delta_w)

            # Update state and action
            s = s_
            a = a_

            if done or n >= 5000:
                #print("Episode finished after %s steps" % (step,))
                times.append(step)
                break

    # Get the genome fitness (avg steps) * 10 = seconds
    fitness = np.mean(times) / 50
    print("Genome {} fitness: {}".format(genome.key,fitness))
    return fitness
    



def run():
    print("[run] Starting main")
    # Get configuration parameters
    config.build(CONFIG_FILE)

    # Set logging
    cart.set_logging(True if 'logging' not in config.params else config.params['logging'])

    # Init the evolution object
    evolution = Evolution(config.params)

    # Run evolution
    winner = evolution.run(fitness_function=eval_genome,generations=config.params['generations'])
    
    # Eval genome
    eval_genome(winner, test=True)

    # Save winner as JSON
    with open(REL_PATH+"/results/winner.json","w") as file:
        json.dump(winner.to_json(), file)

    # Print the net
    # [x],[theta],[phi],[x_dot],[theta_dot],[phi_dot]
    node_names = {-1:'X', -2: 'Theta', -3:'Phi', -4:'X dot', -5:'Theta dot', -6:'Phi dot', 0:'Force'}
    visualize.draw_net(config.params, winner, True, node_names=node_names, filename=REL_PATH+"/results/winner.gv")


def get_caos():
    ''' Run two simulations and get the delta of the initial
        conditions to get the caos
    '''
    cart_1 = DoublePendulumCart()
    cart_2 = DoublePendulumCart()
    cart_1.logging=False
    cart_2.logging=False
    
    # Initial states
    state_1 = np.matrix([[0],[0.01],[0],[0],[0],[0]])
    state_2 = np.matrix([[0],[0.02],[0],[0],[0],[0]])

    # Set states
    cart_1.set_state(state_1)
    cart_2.set_state(state_2)

    # States
    x_1 = []
    theta_1 = []
    phi_1 = []
    x_2 = []
    theta_2 = []
    phi_2 = []
    
    # Actuator
    a = 0

    print("Initial state:")
    print(cart_1.state)
        
    # First simulation
    print("1...")
    for n in range(1,250000):
        #print(1,n)
        # Save angles
        x_1.append(cart_1.state.item(0))
        theta_1.append(cart_1.state.item(1))
        phi_1.append(cart_1.state.item(2))
        # Render
        #cart_1.render()
        # Cart state    
        #print(a)
        s_, r, done, info = cart_1.step(a, actuator=None)
        #print("Cart state: ")
        #print(s_)
        # How many steps did the cart hold?
        s = s_

    # Second simulation
    print("2...")
    for n in range(1,250000):
        #print(2,n)
        # Save angles
        x_2.append(cart_2.state.item(0))
        theta_2.append(cart_2.state.item(1))
        phi_2.append(cart_2.state.item(2))
        # Render
        #cart_2.render()
        # Cart state    
        #print(a)
        s_, r, done, info = cart_2.step(a, actuator=None)
        #print("Cart state: ")
        #print(s_)
        # How many steps did the cart hold?
        s = s_

    with open('{}/results/caos.json'.format(REL_PATH),'w') as file:
        json.dump({
            "x_1" : x_1,
            "theta_1" : theta_1,
            "phi_1" : phi_1,
            "x_2" : x_2,
            "theta_2" : theta_2,
            "phi_2" : phi_2
        }, file)


def test_model():
    ''' Test model behaviour with any external
    '''
    global cart
    
    step = 0
    done = False
    s = cart.reset()
    a = 0
    for n in range(1,5000):

        # Render
        cart.render()

        # Cart state    
        print(a)
        s_, r, done, info = cart.step(a, actuator=None)
        print("Cart state: ")
        print(s_)

        # How many steps did the cart hold?
        step += 1

        s = s_



def test_solution(filename='winner'):
    ''' Load winner from json file and simulate
    '''
    from models.genome import Genome
    # Load file
    config.build(CONFIG_FILE)
    with open('{}/results/{}.json'.format(REL_PATH,filename),'r') as file:
        winner_dict = json.load(file)
    winner = Genome(config.params, winner_dict['key'], rep=winner_dict, init=False)
    print("Evaluating solution...")
    print(winner)
    fit = eval_genome(winner, test=True)
    print("Fitness: "+str(fit))
    node_names = {-1:'X', -2: 'Theta', -3:'Phi', -4:'X dot', -5:'Theta dot', -6:'Phi dot', 0:'Force'}
    #visualize.draw_net(config.params, winner, True, node_names=node_names, filename=REL_PATH+"/results/winner_solution.gv")


if __name__ == '__main__':
    print(sys.argv)
    if 'solution' in sys.argv:
        filename='winner' if len(sys.argv) < 3 else sys.argv[2]
        test_solution(filename)
    elif 'model' in sys.argv:
        test_model()
    elif 'caos' in sys.argv:
        get_caos()
    else:
        run()
        
