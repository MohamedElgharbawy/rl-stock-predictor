from typing import Optional, Union, List
from collections import deque
import gym
import numpy as np
import pandas as pd

from gym.core import RenderFrame

class TradingEnv(gym.Env):
    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def __init__(self, symbol, window_size=10):
        self.curr_window = deque(maxlen=window_size)
        self.num_stocks_owned = 0
        self.starting_amount = 10000
        self.bank_account = self.starting_amount
        self.allowance_per_day = 30
        self.portfolio_value = self.starting_amount

        dataset = pd.read_csv("data/processed/final_dataset.csv")
        self.stock_history = dataset[dataset["Symbol"] == symbol]
        self.stock_history["Date"] = pd.to_datetime(self.stock_history["Date"])
        self.stock_history.sort_values(by="Date")

        self.window_size = window_size
        self.reset()

    def _destroy(self):
        pass

    def _reset(self):
        self.day_index = 0
        self.bank_account = self.starting_amount
        self.portfolio_value = self.bank_account
        self.num_stocks_owned = 0
        for t in range(self.window_size):
            self.curr_window.append(self._get_day(self.day_index).loc["Open"])
            self.day_index += 1
        return np.array(list(self.curr_window))

    def _step(self, action): 
        """Action should be a single number of how much money worth of the stock we should buy or sell at a given timestep.
        Advance the timestep, and get reward based on how much the stock price changes and how much we own"""

        curr_stock_price = self.curr_window[-1]

        next_day = self._get_day(self.day_index)
        next_stock_price = next_day.loc["Open"]

        self.curr_window.popleft()
        self.curr_window.append(next_stock_price)
        next_state = np.array(list(self.curr_window))

        # TODO: Add bank account, reward = starting amount - (bank acc + value of stocks)
        # TODO: NN outputs [-1, 1]. [-1, 0) represents selling 0-100% of stocks, [0, 1] represents buying 0-100% of bank acc worth of stock.
        # TODO: Get daily allowance

        # Allowance
        self.bank_account += self.allowance_per_day

        portfolio_value_without_trading_today = self.num_stocks_owned * next_stock_price + self.bank_account

        # Sell action % of our stocks
        if action < 0 and self.num_stocks_owned > 0:
            num_stocks_to_sell = -1 * action * self.num_stocks_owned
            self.bank_account += num_stocks_to_sell * next_stock_price
            self.num_stocks_owned *= -1 * action

        # Buy action * bank account worth of stock
        if action >= 0:
            value_of_stocks_to_buy = self.bank_account * action
            self.num_stocks_owned += value_of_stocks_to_buy/next_stock_price
            self.bank_account -= value_of_stocks_to_buy

        self.portfolio_value = self.num_stocks_owned * next_stock_price + self.bank_account

        reward = self.portfolio_value - portfolio_value_without_trading_today

        self.curr_stock_price = next_stock_price
        self.day_index += 1

        done = self.day_index == len(self.stock_history.index)

        return next_state, reward, done, None

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def _get_day(self, index):
        return self.stock_history.iloc[index]