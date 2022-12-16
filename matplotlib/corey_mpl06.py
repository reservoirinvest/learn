from pathlib import Path
import pandas as pd

from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# set a default style
plt.style.use("fivethirtyeight")

# file path
p = str(Path(__file__).parent)

data = pd.read_csv(p+r"\histo_data.csv")
ids = data['Responder_id']
ages = data['Age']

# ages = [18, 19, 21, 25, 26, 26, 30, 32, 38, 45, 55]

bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

plt.hist(ages, bins=bins, edgecolor='black', log=True)


median_age = 29
color = '#fc4f30'

plt.axvline(median_age, color=color, label='Age Median', linewidth=2)

plt.legend()

plt.title('Ages of Respondents')
plt.xlabel('Ages')
plt.ylabel('Total Respondents')

plt.tight_layout()

plt.show()
