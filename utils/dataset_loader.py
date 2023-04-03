import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mplfinance as mpf
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from PIL import Image
import logging
logging.basicConfig(
     level=logging.INFO,
     format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
     datefmt='%H:%M:%S'
 )


class CustomDataset(Sequence):

    def __init__(self, data_path, sequence_length, batch_size, return_images=False):
        self.df = pd.read_csv(data_path, parse_dates=True,
                              usecols=["timestamp", "open", "high", "low", "close", "volume"])
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.steps_per_epoch = len(self.df) // self.batch_size
        self.return_images = return_images
        self.style = mpf.make_mpf_style(base_mpf_style='charles', gridcolor='lightgray')

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = start_index + self.sequence_length + self.batch_size
        if end_index > self.df.shape[0]:
            pass 
        # x = self.df.iloc[start_index:end_index][["open"]].values.reshape(self.batch_size, self.sequence_length, -1)
        x = []
        y = []
        for j in range(0, batch_size):
            x.append(self.df.iloc[start_index:start_index+50][["timestamp", "open", "high", "low", "close", "volume"]].values)
            y.append(self.df.iloc[start_index + 50][["timestamp", "open", "high", "low", "close", "volume"]].values)
            start_index += 1
        x = np.array(x)
        y= np.array(y)
        if self.return_images:
            image_arrays = self.plot_image(x)
            return x, y, image_arrays
        else:
            return x, y, None

    def plot_image(self, x):

        batch_array = []
        for i in range(0, x.shape[0]):
            i_df = pd.DataFrame(x[i], columns=["timestamp", "open", "high", "low", "close", "volume"])
            i_df['timestamp'] = pd.to_datetime(i_df['timestamp'])
            i_df = i_df.set_index("timestamp")
            i_df = i_df[["open", "high", "low", "close", "volume"]].astype("float64")
            fig, ax = mpf.plot(i_df, type='candle', style=self.style, volume=True,
                               scale_padding={'left': 0, 'right': 0, 'top': 0.5, 'bottom': 0}, returnfig=True)
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            batch_array.append(data)
            # https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
            plt.close(fig)
        return np.array(batch_array)

    def on_epoch_end(self):
        pass


# Example usage
data_path = 'apple daily data.csv'
sequence_length = 50
batch_size = 32

dataset = CustomDataset(data_path, sequence_length, batch_size, return_images=False)

# Loop over the dataset and display the data
for i, (x, y, images) in enumerate(dataset):
    logging.info("Batch {} - X: {} - Y : {} - image_arrays : {}".format(
        i, x.shape, y.shape, images))
