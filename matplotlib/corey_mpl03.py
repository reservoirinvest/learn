# pie charts
from matplotlib import pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.style.use("fivethirtyeight")

# Language Popularity
slices = [59219, 55466, 47544, 36443, 35917, 31991, 27097,
          23030, 20524, 18523, 18017, 7920, 7331, 7201, 5833]
labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java', 'Bash/Shell/PowerShell',
          'C#', 'PHP', 'C++', 'TypeScript', 'C', 'Other(s):', 'Ruby', 'Go', 'Assembly']
# colors = ['#008fd5', '#fc4f30', '#e5ae37', '#6d904f']

explode = [0, 0, 0, 0.1, 0]

# Make the dataset smaller!
slices = [59219, 55466, 47544, 36443, 35917]
labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java']

plt.pie(slices, labels=labels, wedgeprops={
        'edgecolor': 'black'}, explode=explode, shadow=True, startangle=90,
        autopct='%1.1f%%')

plt.title("My Awesome Pie Chart")
plt.tight_layout()
plt.show()
