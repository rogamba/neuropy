import os
import json
from pprint import pprint

# App directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PATH = os.path.dirname(os.path.realpath(__file__)) + "/"

params = {}

def build(filename):
    global params
    with open(filename) as data_file:
        params = json.load(data_file)
    print("Configuration parameters:")
    pprint(params)
    return params


if __name__=='__main__':
    build()
    pprint(params)