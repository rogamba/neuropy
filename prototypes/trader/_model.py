import logging
import math
import numpy as np
from pprint import pprint

class Trader(object):

    def __init__(self, history=None, pointer=0):

        # Data as a pandas df
        if history:
            # Set data
            self.history = history

        # Init variables
        self.profit = 0             # Initial profit of 0
        self.balance = 100         # Balance in the base currency (btc)
        self.bag = 100             # Balance in the other currency
        self.amt_buy = 1            # 1 Ether
        self.amt_sell = 1           # 1 Ether
        self.commission = 0.00098   # Cost of every transaction
        self.price = 0              # Price of the value you want to exchange
        self.gap = 0.02             # Loss of value due to the gap between buy and sale

        # Stack of all the states in the history
        self.pointer = 0

        # State
        '''
        self.posture = 1
        self.profit = 0
        self.states = []
        self.transactions = []
        self.state = {
            "price" : self.history[0]['price'],
            "timestamp" : self.history[0]['timestamp'],
            "prev_price" : 0,
            "imm_prev_transaction" : 0,
            "prev_transaction" : 0,
            "day" : self.history[0]['day'],
            "hour" : self.history[0]['hour'],
            "minute" : self.history[0]['minute'],
            "posture" : "",
            "balance" : self.balance,
            "bag" : self.balance,
            "value" : self.balance,
            "transaction" : 0
        }

         # First trade by default
        self.trade(1).next()
        '''
        self.reset()


    @staticmethod
    def convert_data(df):
        """ Transform the history df into list
            co compare with the states
        """
        print("Converting history...")
        return [ dict(row) for i, row in df.iterrows() ]


    def reset(self):
        """ Restart states and transactions
        """
        self.state = {
            "price" : self.history[0]['price'],
            "timestamp" : self.history[0]['timestamp'],
            "prev_price" : 0,
            "imm_prev_transaction" : 0,
            "prev_transaction" : 0,
            "day" : self.history[0]['day'],
            "hour" : self.history[0]['hour'],
            "minute" : self.history[0]['minute'],
            "posture" : "",
            "balance" : self.balance,
            "bag" : self.balance,
            "value" : self.balance,
            "transaction" : 0
        }
        self.posture = 1
        self.states = []
        self.transactions = []
        self.balance = 100
        self.bag = 100
        self.pointer = 0
        self.profit = 0
        self.value = 0
        self.initial_value = self.calculate_value()
        self.trade(1).next()


    def calculate_value(self):
        """ Get the current value in the base burrency
            to ceck if we have increased it
        """
        base = self.balance
        bag = self.bag * self.state['price'] * (1-self.gap)
        value = base + bag
        self.value = value
        return value
        

    def traverse_history(self):
        """ Iterate all data
        """
        # For second to length of history
        for i in range(self.pointer, len(self.history)):
            #print("....")
            #print(self.pointer)
            self.pointer = i
            # Update current_state variable
            self.state = {
                "price" : self.history[i]['price'],
                "timestamp" : self.history[i]['timestamp'],
                "day" : self.history[i]['day'],
                "hour" : self.history[i]['hour'],
                "minute" : self.history[i]['minute']
            }
            yield self.history[i]



    def get_vector(self):
        """ From the current state, get the vector that
            will be fed to the NN
        """  
        #print(self.state)
        '''
        print("""
            Price {}
            Last Price {}
            Last Period Transaction {}
            Last Transaction {}
            Las Value {}
            Last day {}
            Last hour {}
            Last minute {}
            --------------
            Balance {}
            Bag {}
        """.format(
            self.state['price'],
            self.states[-1]['price'],
            self.states[-1]['transaction'],
            self.transactions[-1]['transaction'],
            self.value,
            self.state['day'],
            self.state['hour'],
            self.state['minute'], 
            self.balance, 
            self.bag, 
        ))   
        '''       
        self.state_vector = np.array([
            self.state['price'],
            self.states[-1]['price'],
            self.states[-1]['transaction'],
            self.transactions[-1]['transaction'],
            self.value,
            self.state['day'],
            self.state['hour'],
            self.state['minute'],
        ])

        return self.state_vector



    def trade(self, action=None):
        """ Posture can be from -1 to 1
        """
        #print("Trading {}".format(action))
        # Buy
        if action > 0.2 : self.posture = 1
        # Hold
        if action < 0.2 and action > -0.2: self.posture = 0
        # Sell
        if action < -0.2: self.posture = -1
        
        # Evaluate posture and calculare actual cost of trade
        #print("Posture: {}".format(self.posture))
        if self.posture == 1:
            _amt = self.amt_buy
            _base = (_amt * self.state['price'] \
                + (_amt * self.commission)) * -1
        
        elif self.posture == -1:
            _amt = self.amt_sell
            _base = _amt * self.state['price'] \
                + (_amt * self.commission) \
                + (_amt * self.gap)
            _amt = _amt * -1 

        # Set posture to 0 if no balance available
        if (self.posture == 1 and self.balance < abs(_base)) \
            or (self.posture == -1 and self.bag < abs(_amt)):
            print("NOT enough amount!!")
            self.stop=True
            self.posture = 0

        if self.posture == 0:
            _amt = 0
            _base = 0

        # Modify balances
        self.transaction = _base
        self.amt = _amt
        self.balance = self.balance + _base
        self.bag = self.bag + _amt
        self.value = self.calculate_value()
        #print("Posture : {} // Transaction: {}".format(self.posture, self.transaction))

        return self


    def next(self):
        """ Complete state and transaction and append
        """
        #print("Price: {}, Balace: {}, Bag: {}, Total Value: {}, Transaction: {}".format(
        #    self.state['price'],
        #    self.balance,
        #    self.bag,
        #    self.value,
        #    self.transaction
        #))
        # Append into state
        self.state['posture'] = self.posture
        self.state['transaction'] = self.transaction
        self.state['amt'] = self.amt
        self.state['value'] = self.value
        self.state['balance'] = self.balance
        self.state['bag'] = self.bag
        self.states.append(self.state)
        
        # Append into transactions
        if self.posture != 0:
            self.transactions.append({
                "posture" : self.posture,
                "transaction" : self.transaction,
                "amt" : self.amt,
                "value" :self.value,
                "price" : self.state['price'],
                "timestamp" : self.state['timestamp']
            })

        return self


    def status(self):
        """ True if total value hasn't reduced to half 
            of it's initial value
        """
        return  self.value > self.initial_value/2 \
                and self.bag > 0 \
                and self.balance > 1



    def get_profit(self):
        """ Get the profit of the transaction that 
            just happened right before.
        """
        # Profit from previous transactions
        values = [t['value'] for t in self.transactions]

        profits = []
        base = None
        for v in values:
            if not base:
                base = v
            profit = v - base
            profits.append(profit)
            base = v

        return np.array(profits).sum()

        # Get all values to get profit
        #return np.array([ s['value'] for s in self.states ]).mean()


