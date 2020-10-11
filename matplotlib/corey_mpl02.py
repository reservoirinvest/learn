# Bar Charts
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from collections import Counter

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.style.use('fivethirtyeight')

# file path
p = str(Path(__file__).parent)

data = pd.read_csv(p+"\data.csv")
ids = data['Responder_id']
lang_responses = data['LanguagesWorkedWith']

language_counter = Counter()

for response in lang_responses:
    language_counter.update(response.split(';'))

languages = []
popularity = []

for item in language_counter.most_common(15):
    languages.append(item[0])
    popularity.append(item[1])

languages.reverse()
popularity.reverse()

plt.barh(languages, popularity)

plt.title("Most Popular Languages")
# plt.ylabel("Programming Languages")
plt.xlabel("Number of People Who Use")

plt.tight_layout()

plt.show()

# get data from _df_ohlc.pkl
p = str(Path(__file__).parents[2])
df_ohlc = pd.read_pickle(p+r"\data\nse\_df_ohlc.pkl")
