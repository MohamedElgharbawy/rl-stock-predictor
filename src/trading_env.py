import sys, math
import numpy as np
import csv


class TradingEnv: #(gym.Env): # not going to make it a gym env this time, but will use similar conventions
    continuous = True

    def __init__(self, dataset_file):
        
        self.curr_stock_price = None # This is the only part of the state that the env will hold on to. The windown is held only by the trading agent.

        self.curr_step = None
        self._reset()

        csvfile = open(dataset_file, 'r')
        self.csv_reader = csv.reader(csvfile)
        header = self.csv_reader.next()
        print(header)

        first_day = self.csv_reader.next()
        self.curr_stock_price = float(first_day[2]) #MAGIC NUMBER HERE 
        # Note: the data is in the format "Symbol,Date,Open,High,Low,Close,AdjClose,Volume"
        # For now, I am just going to use the opening stock price from day to day (ie. at index 2)

    def _destroy(self):
        pass

    def _reset(self):
        self.curr_step = 0
        return self._step(None)

    def _step(self, action): 
        '''Action should be a single number of how much money worth of the stock we should buy or sell at a given timestep.
        Advance the timestep, and get reward based on how much the stock price changes and how much we own'''
        if action is None: # reset action
            pass
        nextday = self.csv_reader.next()
        next_stock_price = nextday[2]

        num_stocks_to_buy = action / self.curr_stock_price
        reward = (next_stock_price - self.curr_stock_price) * num_stocks_to_buy
      
        self.curr_stock_price = self.next_stock_price

        # todo
        done = None
        info = None

        return np.array([next_stock_price]), reward, done, info

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)