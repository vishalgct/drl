import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=10000, window_size=10):
        self.data = data
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.current_step = window_size

        self.action_space = spaces.Discrete(3)  # Buy, hold, or sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(window_size,))

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = self.window_size
        return self.data[self.current_step - self.window_size:self.current_step]

    def step(self, action):
        self.current_step += 1

        if self.current_step >= len(self.data):
            done = True
        else:
            done = False

        if action == 0:  # Buy
            self.balance -= self.data[self.current_step]
        elif action == 1:  # Hold
            pass
        elif action == 2:  # Sell
            self.balance += self.data[self.current_step]

        reward = self.balance
        state = self.data[self.current_step - self.window_size:self.current_step]

        return state, reward, done, {}
