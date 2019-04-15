import logging
import math
import numpy as np
from pprint import pprint


class Trader(object):

    """ 
        Trader V2.0
        @State variables:
        : Price
        : Change of price (difference vs previous price)
        : Immediate previous posture (sell, buy, hold) - (-1, 0, 1)
        : Previous successful transaction (-1, 1)
        : Accumulate minutes from the previous successful transaction <int>
        : Minute of the day??
    """

    def __init__(self, history=None, pointer=0):

        # Data as a pandas df
        if history:
            # Set data
            self.history = history

        # Init variables
        self.profit = 0             # Initial profit of 0
        self.initial_balance = 100  # Balance in the base currency (btc)
        self.initial_bag = 100      # Balance in the other currency (etc, xrp, etc...)
        self.initial_value = 100
        self.amt_buy = 1            # 1 Ether
        self.amt_sell = 1           # 1 Ether
        self.price = 0              # Price of the value you want to exchange
        self.gap = 0 #0.02             # Loss of value due to the gap between buy and sale    <<<< Change??
        self.commission = 0 #0.00098   # Cost of every transaction

        # Pointer to the current moment in history
        self.i = 1
        self.j = 0

        self.reset()


    @staticmethod
    def convert_data(df):
        """ Transform the history df into list
            to compare with the states
        """
        print("Converting history...")
        return [ dict(row) for i, row in df.iterrows() ]


    def reset(self):
        """ Restart states and transactions
        """
        self.transactions = []
        self.balance = self.initial_balance
        self.bag = self.initial_bag
        self.state = {
            "price" : self.history[0]['price'],
            "timestamp" : self.history[0]['timestamp'],
            "prev_price" : 0,
            "imm_prev_transaction" : 0,
            "prev_transaction" : 0,
            "minutes_from_prev_trans" : 0,
            "price_from_prev_trans" : 0,
            "day" : self.history[0]['day'],
            "hour" : self.history[0]['hour'],
            "minute" : self.history[0]['minute'],
            "posture" : "",
            "balance" : self.balance,
            "bag" : self.balance,
            "transaction" : 0
        }
        self.value = self.calculate_value()
        self.state['value'] = self.value
        self.posture = 1
        self.states = []
        self.i = 1
        self.profit = 0
        self.trade(1).next()


    def calculate_value(self):
        """ Get the current value in the base currency
            to ceck if we have increased it
        """
        price = self.state['price']
        base = self.balance
        bag_value = (self.bag * price) * (1-self.gap)
        value = base + bag_value
        self.value = value
        return value
        

    def traverse_history(self):
        """ Iterate all historic data
        """
        # For second to length of history
        for i in range(self.i, len(self.history)):
            #print("....")
            #print(self.pointer)
            self.i = i
            # Update current_state variable
            self.state = {
                "price" : self.history[i]['price'],
                "timestamp" : self.history[i]['timestamp'],
                "day" : self.history[i]['day'],
                "hour" : self.history[i]['hour'],
                "minute" : self.history[i]['minute'],
                "minute_of_the_day" : (self.history[i]['hour']*60) + self.history[i]['minute'],
                "diff_from_prev_state" : self.get_diff_from_prev_state(),
                "diff_from_prev_trans" : self.get_diff_from_prev_trans(),
                "minutes_from_prev_trans" : self.get_mins_from_prev_trans(),
                "immediate_prev_trans" :  self.get_immediate_prev_trans(),
                "successful_prev_trans" :  self.get_successful_prev_trans()
            }
            yield self.history[i]


    def get_vector(self):
        """ From the current state, get the vector that
            will be fed to the NN
            - Price
            - Change of price from prev state (difference vs previous price)
            - Change of price from last transaction (difference vs previous price)
            - Accumulate minutes from the previous successful transaction <int>
            - Immediate previous transaction 
            - Previous successful transaction 
            - Minute of the day
        """  
        self.state_vector = np.array([
            self.state['price'],
            self.state['diff_from_prev_state'],    
            self.state['diff_from_prev_trans'],
            self.state['minutes_from_prev_trans'],
            self.state['immediate_prev_trans'],
            self.state['successful_prev_trans'],
            self.state['minute_of_the_day'],
        ])

        #print(self.state)
        '''
        print("""
            Current Price {}
            Change od price {}
            Immediate previous posture {}
            Previous successful transaction {}
            Minutes from prev transaction {}
            Minute {}
            --------------
            Balance {}
            Bag {}
        """.format(
            self.state['price'],
            self.state['price']-self.states[-1]['price'],
            self.state['minutes_from_prev_trans'],
            self.states[-1]['transaction'],
            self.transactions[-1]['transaction'],
            self.state['minute'],
            self.balance, 
            self.bag, 
        ))   
        '''  
             
        return self.state_vector



    def trade(self, action=None):
        """ Posture can be from -1 to 1
        """
        # Evaluate network output
        # Buy
        if action >= 0.2 : 
            #print("BUY")
            self.posture = 1
        # Hold
        elif action < 0.2 and action > -0.2: 
            #print("HOLD")
            self.posture = 0
        # Sell
        elif action <= -0.2: 
            #print("SELL")
            self.posture = -1
        # Nan
        else:
            #print("NaN") 
            self.posture = 0
        
        # Evaluate posture and calculare actual cost of trade
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

        if self.posture == 0:
            _amt = 0
            _base = 0

        # Modify balances
        self.transaction = _base
        self.amt = _amt
        self.balance = self.balance + _base
        self.bag = self.bag + _amt
        self.value = self.calculate_value()

        #print("[Transaction] Price:{}, Transaction: {} ".format(self.state['price'],_base))

        return self


    def get_immediate_prev_trans(self):
        """ Return cost of the previous state transaction
            <float> (buy, sell or hold)
        """
        return self.states[-1]['transaction']

    def get_successful_prev_trans(self):
        """ Return total cost of last successful transaction 
            <float> (either buy or sell)
        """
        if len(self.transactions) > 0:
            return self.transactions[-1]['transaction']
        else:
            return 0

    def get_mins_from_prev_trans(self):
        """ Time diff in mins between i and j
        """
        prev = self.history[self.j]['timestamp']
        current = self.history[self.i]['timestamp']
        return (current-prev)/60

    def get_diff_from_prev_trans(self):
        """ Time diff in mins between i and j
        """
        return self.history[self.i]['price'] - self.history[self.j]['price']

    def get_diff_from_prev_state(self):
        """ Price difference of current price vs 
            the price of the last transaction
        """
        return self.history[self.i]['price'] - self.history[self.i-1]['price']


    def get_unitary_profit(self):
        """ If I'm selling, calculate the diff when last time
            I bough
        """
        if self.posture == -1:
            u_profit = self.history[self.i]['price']-self.history[self.j]['price']
            #print("Bought at {}, sold at {}. Profit: {}".format(
            #    self.history[self.j]['price'],
            #    self.history[self.i]['price'],
            #    u_profit,
            #))
        else:
            u_profit = 0
        return u_profit



    def next(self):
        """ Complete state and transaction and append
        """
        #print()
        #print("Balance: {}, Bag: {}, Value: {}".format(self.balance, self.bag, self.value))
        
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
            # Append transactions
            self.transactions.append({
                "posture" : self.posture,
                "transaction" : self.transaction,
                "amt" : self.amt,
                "value" :self.value,
                "price" : self.state['price'],
                "timestamp" : self.state['timestamp'],
                "minutes_from_prev_trans" : self.state['minutes_from_prev_trans'],
                "unitary_profit" : self.get_unitary_profit(),
                "pointer" : self.i
            })
            # Set j, pointer to last transaction
            self.j = self.i

        return self


    def status(self):
        """ True if total value hasn't reduced to half 
            of it's initial value
        """
        return  self.value > self.initial_value/2 \
                and self.bag > 0 \
                and self.balance > 1


    def get_overall_unitary_profit(self):
        """ Get the profit discarding the amount of
            coins in the bag
        """
        # Get profits
        profits = [t['unitary_profit'] for t in self.transactions]
        return np.array(profits).sum()


    def get_overall_profit(self):
        """ Get the total profit to calculate the
            fitness of the individual
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



