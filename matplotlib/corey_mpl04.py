# Stacked plots

from matplotlib import pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.style.use("fivethirtyeight")

minutes = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# developer1 = [1, 2, 3, 3, 4, 4, 4, 4, 5]
# developer2 = [1, 1, 1, 1, 2, 2, 2, 3, 4]
# developer3 = [1, 1, 1, 2, 2, 2, 3, 3, 3]

# fading out of developers ... from 8 hours
developer1 = [8, 6, 5, 5, 4, 2, 1, 1, 0]
developer2 = [0, 1, 2, 2, 2, 4, 4, 4, 4]
developer3 = [0, 1, 1, 1, 2, 2, 3, 3, 4]

labels = ['developer1', 'developer2', 'developer3']
colors = ['#6d904f', '#008fd5', '#e5ae37']

# plt.pie([1, 1, 1], labels=["developer 1", "developer2", "developer3"])
plt.stackplot(minutes, developer1, developer2,
              developer3, labels=labels, colors=colors)
plt.legend(loc=(0.07, 0.05))  # to show the labels in the legend

plt.title("My Awesome Stack Plot")
plt.tight_layout()
plt.show()

# Colors:
# Blue = #008fd5
# Red = #fc4f30
# Yellow = #e5ae37
# Green = #6d904f
