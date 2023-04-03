import pandas as pd
import mplfinance as mpf
import requests
import io
import numpy as np
import matplotlib.pyplot as plt

# Define Alpha Vantage API parameters
symbol = 'AAPL'
api_key = '2HH9NL37TSD5ZUOO'

# Define the API endpoint
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}&datatype=csv&outputsize=full'

# Fetch stock data from the API endpoint and store it in a DataFrame
# response = requests.get(url)
# df = pd.read_csv(io.StringIO(response.text), index_col='timestamp', parse_dates=True)

df = pd.read_csv("apple daily data.csv", index_col='timestamp', parse_dates=True)

# Define the plot style
style = mpf.make_mpf_style(base_mpf_style='charles', gridcolor='lightgray')

# Plot the candlestick chart
# mpf.plot(df, type='candle', style=style, volume=True, ylabel='Price ($)', mav=(5, 10, 20))

for i in range(0, len(df), 100):
    fig, ax = mpf.plot(df.iloc[i:i+100], type='candle', style=style, volume=True, ylabel='Price ($)', mav=(5, 10, 20), returnfig=True)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    print(data.shape)

print("Hello")
# mpf.plot(df.iloc[0:200], type='candle', style=style, volume=True, ylabel='Price ($)',
#          scale_padding={'left': 0, 'right': 0, 'top': 0, 'bottom': 0}, savefig=dict(fname='image.png',dpi=100,pad_inches=0.25))
# Save the chart as an image
# plt.savefig('image.png')

