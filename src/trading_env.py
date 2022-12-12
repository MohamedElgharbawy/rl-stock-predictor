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
        self.portfolio_value = 0

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

        # TODO: Determine how we want to handle a limit of buying stocks, for now 1 = one stock
        num_stocks_to_buy = action
        if action < 0 and abs(action) > self.num_stocks_owned:
            # Clip and sell everything
            num_stocks_to_buy = -self.num_stocks_owned

        self.num_stocks_owned += num_stocks_to_buy
        reward = (next_stock_price - curr_stock_price) * self.num_stocks_owned

        self.curr_stock_price = next_stock_price
        self.day_index += 1

        done = self.day_index == len(self.stock_history.index)

        self.portfolio_value = self.num_stocks_owned * next_stock_price

        return next_state, reward, done, None

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def _get_day(self, index):
        return self.stock_history.iloc[index]