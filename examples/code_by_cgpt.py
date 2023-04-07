import gym
import numpy as np
import pandas as pd
import yfinance as yf
from gym import spaces
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import Sequential
from alpha_vantage.timeseries import TimeSeries
import io
import logging
logging.basicConfig(
     level=logging.INFO,
     format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
     datefmt='%H:%M:%S'
 )

from alpha_vantage.timeseries import TimeSeries

def download_data(stock_name, start_date, end_date, api_key):
    # ts = TimeSeries(key=api_key, output_format='pandas')
    # stock_data, _ = ts.get_daily_adjusted(stock_name, outputsize='full')
    # stock_data = stock_data.loc[start_date:end_date]
    # stock_data.to_csv("apple daily data from API.csv")
    stock_data = pd.read_csv("apple daily data from API.csv")
    stock_data = stock_data[['4. close', '5. adjusted close', '6. volume', '7. dividend amount', '8. split coefficient']]
    stock_data.columns = ['Close', 'Adj Close', 'Volume', 'Dividend', 'Split']
    return stock_data

api_key = '2HH9NL37TSD5ZUOO'
stock_name = 'AAPL'
start_date = '2015-01-01'
end_date = '2021-01-01'
stock_data = download_data(stock_name, start_date, end_date, api_key)

class TradingEnvironment(gym.Env):
    def __init__(self, stock_data):
        super(TradingEnvironment, self).__init__()

        self.stock_data = stock_data
        self.max_steps = len(self.stock_data) - 1
        self.current_step = 0

        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(low=0, high=1, shape=(5, 1), dtype=np.float32)

        self.scaler = MinMaxScaler()

    def reset(self):
        self.current_step = 0
        self.total_profit = 0
        self.current_price = self.stock_data.iloc[self.current_step]['Close']
        self.position = None
        self.stock_data_scaled = self.scaler.fit_transform(self.stock_data)

        return self._get_observation()

    def step(self, action):
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        else:
            done = False

        self.current_price = self.stock_data.loc[self.current_step, 'Close']

        if action == 0:  # Buy
            if self.position is None:
                self.position = self.current_price
        elif action == 1:  # Sell
            if self.position is not None:
                profit = self.current_price - self.position
                self.total_profit += profit
                self.position = None

        obs = self._get_observation()
        reward = self.total_profit
        info = {}

        return obs, reward, done, info

    def _get_observation(self):
        return self.stock_data_scaled[self.current_step].reshape(5, 1)

def create_q_network(input_shape, action_space):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
    return model

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # Size of the state space.
        self.action_size = action_size  # Size of the action space.
        self.memory = []  # A list to store the agent's experiences (state, action, reward, next_state, done).
        self.gamma = 0.95 # discount factor for future rewards
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # The minimum value of epsilon to ensure some exploration continues.
        self.epsilon_decay = 0.995  # The decay factor for epsilon, which reduces epsilon after each episode.
        self.model = create_q_network(state_size, action_size)  # The primary Q-Network for predicting Q-values.
        self.target_model = create_q_network(state_size, action_size)  # The target Q-Network used for calculating target Q-values during training.
        self.update_target_model()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)

            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self, name):
        self.model.save_weights(name)

import random

env = TradingEnvironment(stock_data)
state_size = env.observation_space.shape
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
batch_size = 32
num_episodes = 100

for e in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size[0], state_size[1]])
    total_profit = 0

    for _ in range(env.max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size[0], state_size[1]])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_profit += reward

        if done:
            logging.info(f"Episode: {e + 1}/{num_episodes}, Total Profit: {total_profit}")
            break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    if (e + 1) % 10 == 0:
        agent.update_target_model()

agent.save("trading_agent.h5")

