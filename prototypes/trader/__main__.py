from .model import Trader
import pandas as pd
import os
import sys
import config
from models.evolution import Evolution
from models.phenotype import Phenotype
try:
    from models import rendering
    render=True
except:
    render=False
import models.visualize as visualize
import json

"""
    Trader:
    - Empiezas con un monto de lana 
    - Si se lo acaba o llega al límite inferior se termina la simulación
    - El fitness dependerá de la utilidad obtenida
    - Definir periodos de tiempo (5 minutos mínimo)
    - Cada periodo de tiempo el algoritmo eligirá: buy, sell, hold
"""

REL_PATH = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = REL_PATH+'/config.json'
history = None
historic_data = []
evolution = None 


evals_total = 0
evals_counter = 0


def eval_genome(genome):
    """ 
        Reglas de evaluación: tomar una postura (compra, venta o hold)

        - Fitness: utilidad total de la evaluación
        - Empezar evaluación con X fondo de bitcoins y Y de <coin>
        - Variable de entrada de la red:
          * Precio actual
          * Precio anterior
          * Transacción periodo anterior
          * Transacción anterior
          * Valor total
          * Día
          * Hora
          * Minuto
        - Variables de salida de la red
          * postura: buy, sell, hold
        - Tomar postura de venta al comienzo (compra de x cantidad de )
    """
    global historic_data, evals_total, evals_counter
    # Create phenotype from the genome
    coin = config.params['coin']
    phenotype = Phenotype(config.params, genome)
    net = phenotype.create()

    postures = []
    c = 0

    # Number of evaluations
    trader = Trader(historic_data)
    for e in range(config.params['evals_per_genome']):

        #print("--------------------------------")
        #print("--------------------------------")
        #print("--------------------------------")
        evals_counter+=1
        print("Evaluation {} of {}".format(
            evals_counter,
            evals_total
        ))
        trader.reset()
        posture = None

        for state in trader.traverse_history():
            # Get state vector to feed the net
            _svector = trader.get_vector()
            #print("State Vector {}".format(_svector))
            output = net.activate(_svector)
            #print("Output: {}".format(output))

            # With the posture, trade
            #val_start = trader.calculate_value()            
            action = output[0]
            trader.trade(action)
            #val_end = trader.calculate_value()
            #print("Values: {} to {}".format(val_start, val_end))

            # Status
            status = trader.status()

            if not status or pd.isnull(action):
                print("Breaking, low value, balance or bag...")
                break

            # Reset value
            trader.next()

        unit_profit = trader.get_overall_unitary_profit()
        overall_profit = trader.get_overall_profit()
        fitness = unit_profit

        print("UProfit: {} // Overall Profit: {}".format(
            unit_profit,
            overall_profit
        ))

        print(">>>>>>>>>>>>>>> ")
        print("Fitness: {}".format(fitness))

    if fitness < 0:
        fitness = 0

    return fitness



def load_history(coin):
    global history, historic_data, evolution
    print("Loading history...")
    history = pd.read_csv('{}/data/{}.csv'.format(REL_PATH,coin))
    # Filter just one month
    _history = history
    _history['fulldate'] = _history.apply(
        lambda x: int("{}{}{}".format(
            int(x['year']),
            str(int(x['month'])).zfill(2),
            str(int(x['day'])).zfill(2)
        )),
        axis=1
    )
    # Read after the bubble burst
    _history[_history['fulldate'] > 20180210]
    historic_data = Trader.convert_data(_history)



def run():
    print("[run] Starting trading prototype")
    # Get configuration parameters
    global history, historic_data, evolution, evals_total
    config.build(CONFIG_FILE)

    # Historic data for evolution
    coin = config.params['coin']

    # Load historic data
    load_history(coin)

    # Init the evolution object
    evolution = Evolution(config.params)

    # Run evolution
    evals_total = config.params['generations'] * config.params['pop_size']
    winner = evolution.run(
        fitness_function=eval_genome,
        generations=config.params['generations']
    )
    
    # Eval genome
    eval_genome(winner)

    # Save winner as JSON
    with open(REL_PATH+"/results/winner.json","w") as file:
        json.dump(winner.to_json(), file)

    '''
	    price
	    prev_price
        immediate_prev_transaction (0: hold, +: compra, -: venta)
	    prev_transaction (-: venta, +:compra)
	    total_value (in terms of case_currency)
        día del mes
        hora
        minuto del día
    '''

    # Print the net
    node_names = {
        -1: 'Price', 
        -2: 'Prev_Price', 
        -3: 'Immediate_Prev_Transaction', 
        -4: 'Prev_Transaction', 
        -5: 'Total_Value', 
        -6: 'Day', 
        -7: 'Hour', 
        -8: 'Minute', 
        0:  'Posture'
    }
    if render:
        visualize.draw_net(config.params, winner, True, node_names=node_names, filename=REL_PATH+"/results/winner.gv")


def test_solution():
    ''' Load winner from json file and simulate
    '''
    from models.genome import Genome
    # Load file
    config.build(CONFIG_FILE)
    coin = config.params['coin']
    load_history(coin)
    with open(REL_PATH+'/results/winner.json','r') as file:
        winner_dict = json.load(file)
    winner = Genome(config.params, winner_dict['key'], rep=winner_dict, init=False)
    print("Evaluating solution...")
    print(winner)
    fit = eval_genome(winner)
    print("Fitness: "+str(fit))
    node_names = {
        -1: 'Price', 
        -2: 'Prev_Price', 
        -3: 'Immediate_Prev_Transaction', 
        -4: 'Prev_Transaction', 
        -5: 'Total_Value', 
        -6: 'Day', 
        -7: 'Hour', 
        -8: 'Minute', 
        0:  'Posture'
    }
    visualize.draw_net(config.params, winner, True, node_names=node_names, filename=REL_PATH+"/results/winner_solution.gv")



if __name__ == '__main__':
    if 'solution' in sys.argv:
        test_solution()
    elif 'model' in sys.argv:
        test_model()
    else:
        run()
        