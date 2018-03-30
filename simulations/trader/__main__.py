from evolution import Evolution
from phenotype import Phenotype
import config
import visualize
import os
import json

''' Steps to use the library:
    1. Set the configuration parameters
    2. Import the configuration variables
    3. Set the fitness function you would like to eval your genomes with
    4. Pass the config dict to the evolution instance
    5. Run the evolution process passing the fitness function
        
'''

REL_PATH = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = REL_PATH+'/config.json'

# Hols boundy variation:
hold_boundry = 0.3
dataset = []

coins = [
    ('bitcoin', 'BTC'),
    ('ethereum', 'ETH'),
    ('ripple', 'XRP'),
    ('litecoin', 'LTC')
]

def read_data():
    global dataset
    dataset = pd.read_csv('../data/coin_prices.csv')

def eval_genome(genome):
    ''' Evaluate the genome with the complete set of historical data of the currencies
    '''
    global dataset
    # Create phenotype from the genome
    phenotype = Phenotype(config.params, genome)
    net = phenotype.create()

    fitness = 0
    price_prev = 0 
    data_length = len(dataset)

    for i,point in dataset.iterrows():


        # Iterate every coin to build vector BTC, ETH, RIPPLE
        inputs = []
        for name, coin in coins:
            price = point[coin]
            price_diff = 0 if i == 0 else (price - price_prev)
            prev_price[coin] = price
            


        
        # Get the inputs of the net: actual point, difference with previous point
        price = (point['open'] + point['close']) / 2
        price_diff = 0 if i == 0 else (price - price_prev)
        price_prev = price
        inputs = (price, price_diff)

        # Evaluate every point of the dataset
        o = net.activate(inputs)
        output = o[0]

        # Check profitability of the bet
        if i >= data_length:
            break
        
        # Future price
        future_price = (point['open'] + point['close']) / 2
        future_price_diff = future_price - price
        future_price_diff_percent = (future_price_diff/future_price)*100
        
        # Recommended postures if
        if future_price_diff_percent < 0.5 and future_price_diff_percent > 0.5:
            posture = 'hold'
        elif future_price_diff_percent >= 0.5:
            posture = 'buy'
        elif future_price_diff_percent <= 0.5:
            posture = 'sell'

        print("Price: \t %s" % (price,))
        print("Diff: \t %s" % (price_diff,))
        print("Output: \t %s" % (output,))
        print("Future: \t %s" % (future_price_diff_percent,))
        print("Posture: \t %s" % (posture,))


        # If -0.3 < output < 0.3 : price won't vary enough 
        if output > -0.3 and output < 0.3 and posture == 'hold':
            fitness+=1

        # If 0.3 < output <= 1 : price will raise
        elif output <= -0.3 and posture == 'sell':
            fitness+=1

        # If -1 < output < -0.3 : price will drop
        elif output >= 0.3 and posture == 'buy':
            fitness+=1

    return fitness
        


def run():
    print("[run] Starting main")
    # Get configuration parameters
    config.build(CONFIG_FILE)

    # Load dataset
    read_data()

    # Init the evolution object
    evolution = Evolution(config.params)

    # Run evolution
    winner = evolution.run(fitness_function=eval_genome,generations=config.params['generations'])

    print('\nOutput:')
    ph = Phenotype(config.params, winner)
    winner_net = ph.create()
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    # Print the net
    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config.params, winner, True, node_names=node_names, filename=REL_PATH+"/plots/winner.gv")



if __name__ == '__main__':
    run()
