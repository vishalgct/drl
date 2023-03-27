import numpy as np
import yfinance as yf
from trading_environment import TradingEnvironment
from dqn_agent import DQNAgent

def train_dqn_agent(data, window_size=10, episodes=100, batch_size=32):
    env = TradingEnvironment(data, window_size=window_size)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        print("Episode: {}/{}, Balance: {:.2f}".format(e + 1, episodes, env.balance))

    agent.save("dqn_trader.h5")

symbol = "AAPL"
start_date = "2010-01-01"
end_date = "2021-01-01"

data = yf.download(symbol, start=start_date, end=end_date)
prices = data['Close'].values

# Normalize the data
prices = (prices - np.mean(prices)) / np.std(prices)

train_dqn_agent(prices, window_size=10, episodes=100, batch_size=32)