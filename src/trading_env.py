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
        self.stock_price_window = deque(maxlen=window_size)
        self.company_history_window = deque(maxlen=window_size)
        self.num_stocks_owned = 0
        self.starting_amount = 10000
        self.bank_account = self.starting_amount
        self.allowance_per_day = 30
        self.portfolio_value = self.starting_amount

        dataset = pd.read_csv("data/processed/stock_history_final_dataset.csv")
        self.stock_history = dataset[dataset["Symbol"] == symbol]
        self.stock_history["Date"] = pd.to_datetime(self.stock_history["Date"])
        self.stock_history.sort_values(by="Date")
        
        company = pd.read_csv("data/processed/finance_ratios_final_dataset.csv")
        self.company_history = company[company["Symbol"] == symbol]
        self.company_history["Date"] = pd.to_datetime(self.company_history["period_end_date"])
        self.company_history.sort_values(by="Date")

        self.window_size = window_size
        self.reset()

    def _destroy(self):
        pass

    def _reset(self):
        self.day_index = 0
        self.quarter_index = 0
        self.bank_account = self.starting_amount
        self.portfolio_value = self.starting_amount
        self.num_stocks_owned = 0
        for t in range(self.window_size):
            day = self._get_day(self.day_index)
            day_date = day.loc["Date"]
            quarter = self._get_quarter(self.quarter_index)
            quarter_date = quarter.loc["Date"]
            while quarter_date.year < day_date.year or quarter_date.month < day_date.month:
                self.quarter_index += 1
                quarter = self._get_quarter(self.quarter_index)
                quarter_date = quarter.loc["Date"]

            while day_date.year < quarter_date.year or quarter_date.month < quarter_date.month:
                self.day_index += 1
                day = self._get_day(self.day_index)
                day_date = day.loc["Date"]

            self.stock_price_window.append(day.loc["Open"])
            self.quarter = quarter.drop(labels=["Symbol", "Date", "period_end_date"]).to_numpy(dtype=np.float64)
            self.day_index += 1
        return np.array(list(self.stock_price_window)), np.array(list(self.quarter))

    def _step(self, action): 
        """Action should be a single number of how much money worth of the stock we should buy or sell at a given timestep.
        Advance the timestep, and get reward based on how much the stock price changes and how much we own"""

        next_day = self._get_day(self.day_index)
        next_stock_price = next_day.loc["Open"]

        quarter = self._get_quarter(self.quarter_index)

        self.stock_price_window.popleft()

        self.stock_price_window.append(next_stock_price)
        self.quarter = quarter.drop(labels=["Symbol", "Date", "period_end_date"]).to_numpy(dtype=np.float64)

        next_stock_state = np.array(list(self.stock_price_window))
        next_finance_state = np.array(list(self.quarter))

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

        day_date = next_day.loc["Date"]
        quarter_date = quarter.loc["Date"]

        while quarter_date.year < day_date.year or quarter_date.month < day_date.month:
            self.quarter_index += 1
            quarter = self._get_quarter(self.quarter_index)
            quarter_date = quarter.loc["Date"]

        done = self.day_index == len(self.stock_history.index) or self.quarter_index == len(self.company_history.index)

        return next_stock_state, next_finance_state, reward, done, None

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def _get_day(self, index):
        return self.stock_history.iloc[index]

    def _get_quarter(self, index):
        return self.company_history.iloc[index]