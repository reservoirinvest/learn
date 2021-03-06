import pandas as pd
from matplotlib import pyplot as plt

from pandas.plotting import register_matplotlib_converters

from pathlib import Path
p = str(Path(__file__).parent)  # file path

plt.style.use('seaborn')

data = pd.read_csv(p+r'\subplots_data.csv')
ages = data['Age']
dev_salaries = data['All_Devs']
py_salaries = data['Python']
js_salaries = data['JavaScript']

# using normal staple plt

# plt.plot(ages, py_salaries, label='Python')
# plt.plot(ages, js_salaries, label='JavaScript')

# plt.plot(ages, dev_salaries, color='#444444',
#          linestyle='--', label='All Devs')

# plt.legend()

# plt.title('Median Salary (USD) by Age')
# plt.xlabel('Ages')
# plt.ylabel('Median Salary (USD)')

# to plot in one figure
# fig, (ax1, ax2) = plt.subplots(
#     nrows=2, ncols=1, sharex=True)

# to plot in two figures
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

ax1.plot(ages, dev_salaries, color='#444444',
         linestyle='--', label='All Devs')
ax2.plot(ages, py_salaries, label='Python')
ax2.plot(ages, js_salaries, label='JavaScript')

ax1.legend()
ax1.set_title('Median Salary (USD) by Age')
ax1.set_xlabel('Ages')
ax1.set_ylabel('Median Salary (USD)')

ax2.legend()
ax2.set_title('Median Salary (USD) by Age')
ax2.set_xlabel('Ages')
ax2.set_ylabel('Median Salary (USD)')

plt.tight_layout()

plt.show()

# saving to file system
# fig1.savefig('fig1.png')
# fig2.savefig('fig2.png')
