import sys, math
import numpy as np

import gym

class TradingEnv: #(gym.Env): # not going to make it a gym env this time, but will use similar conventions
    continuous = True

    def __init__(self, dataset):
        
        self.num_stocks_held = 0
        self.stock_prices = dataset

        self.viewer = None
        self.curr_step = None
        self._reset()

    def _destroy(self):
        pass

    def _reset(self):
        self.curr_step = 0
        # reset stuff here
        return self._step(NOOP)[0]

    def _step(self, action): 
        '''Action should be a single number of how much value of the stock we should buy or sell at a given timestep.
        Advance the timestep, and get reward based on how much the stock price changes and how much we own'''

        stocks_to_buy = action / 
        reward = 0
        
        state = self.

        return np.array(state), reward, done, info

    def reset(self):
        return self._reset()

    def step(self, *args, **kwargs):
        return self._step(*args, **kwargs)

if __name__=="__main__":
    #env = LunarLander()
    s = env.reset()
    total_reward = 0
    steps = 0
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        env.render()
        total_reward += r
        if steps % 20 == 0 or done:
            print(["{:+0.2f}".format(x) for x in s])
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: break
