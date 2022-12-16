import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from matplotlib.animation import FuncAnimation

from pathlib import Path
p = str(Path(__file__).parent)  # file path

plt.style.use('fivethirtyeight')

# x_vals = [0, 1, 2, 3, 4, 5]
# y_vals = [0, 1, 3, 2, 3, 5]

# plt.plot(x_vals, y_vals)

x_vals = []
y_vals = []

index = count()


def animate(i):
    # x_vals.append(next(index))
    # y_vals.append(random.randint(0, 5))

    data = pd.read_csv(p+r"\data.csv")
    x = data['x_value']
    y1 = data['total_1']
    y2 = data['total_2']

    plt.cla()  # clear the previous lines

    plt.plot(x, y1, label='Channel 1')
    plt.plot(x, y2, label='Channel 2')

    plt.legend(loc='upper left')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()
