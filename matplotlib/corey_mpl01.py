# Learning Matplotlib from Corey Schafer - https://www.youtube.com/watch?v=UO98lJQ3QGI
# Chapter 1: Simple line charts

import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# get data from _df_ohlc.pkl
p = str(Path(__file__).parents[2])
df_ohlc = pd.read_pickle(p+r"\data\nse\_df_ohlc.pkl")

# print(plt.style.available)
plt.style.use('fivethirtyeight')  # to use the styles
# plt.xkcd()  # comic style!

# reset index to remove (symbol, dte) tuple
df = df_ohlc.reset_index()

# filter out one symbol
df = df[df.symbol == 'AXISBANK']

# set the x-axis data
x = df.date

# print(df_ohlc)

# plt.plot_date(x, df.close, color='black',
#               linestyle='-', marker='', label="close", linewidth=3)
# plt.plot_date(x, df.average, color='blue',
#               linestyle=':', marker='', label="average")

plt.plot(x, df.close, marker='o', label="close")
# plt.plot(x, df.average,
#          label="average")

plt.xlabel('date')
plt.ylabel('prices')
plt.title('AXISBANK')
plt.legend()
plt.tight_layout()  # to add some padding
# plt.grid(True) # no need for grid when a style is used

# save the plots
# plt.savefig(p+r'\ref\learn\plot.png')
# plt.savefig(p+r'\ref\learn\plot.svg', format='svg')

plt.show()
