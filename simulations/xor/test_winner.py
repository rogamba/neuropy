import config
import visualize
from genome import Genome
import os
import json
from pprint import pprint

REL_PATH = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = REL_PATH+'/config.json'
config.build(CONFIG_FILE)

# Load winner object
def load_winner():
    winner = {}
    # Load file
    with open(REL_PATH+'/winner.json','r') as file:
        winner = json.load(file)
    return winner

def run():
    winner = load_winner()
    genome = Genome(config.params, winner['key'], rep=winner)
    print(genome)
    # Simulate genome


if __name__ == '__main__':
    run()