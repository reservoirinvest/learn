# Learning Plotting using matplotlib

## Start
Importing matplotlib does not directly seem to work in py files with data from pandas. So the following code is needed:

<code><pre>
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

\# set a default style
plt.style.use("fivethirtyeight")
</pre></code>

To get data files in the same directory the following code is needed:
<pre><code>from pathlib import Path
p = str(Path(__file__).parent) # file path

data = pd.read_csv(p+r"\fills_data.csv")
</code></pre>

To get path for datafiles from outside, say pickle files, the following code is needed:
<pre><code>from pathlib import Path
df_ohlc = pd.read_pickle(str(Path(__file__).parents[2])+r'\data\nse\_df_ohlc.pkl')
</code></pre>

## Workflow for plotting

Ref: [Matplotlib Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf)

Process comprises:
 - Data Prepration
 - Plot / Sub-plot Creation
 - Plotting
 - Customizing
 - Saving
 - Showing
 - Closing / Clearing

## Notes from 1st chapter (line plots)

1) Format strings are of the syntax <pre>[fmt = '[marker][line][color]'](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html)</pre>
2) Don't use fmt. Use keywords instead! Like:
<pre>
plt.style.use('fivethirtyeight')  # using a style
</pre>
3) Use readymade styles - without grid or setting colour, but setting marker and linestyle
<pre>
plt.plot(x, df.close, marker='o', color="black", marker="o", label="close")
</pre>

## Notes from 2nd chapter (bar plots)
1) One bar plot can be mixed with line (plot) chart
2) Shifting of x-axis is needed for bar plots to make them side-by-side, as follows:
<pre><code># for 3 y-axis bars
import numpy as np
width = 0.25
x_indexes = np.arange(len(ages_x))
plt.bar(x_indexes - width, y1-axis-vals, color...)
plt.bar(x_indexes, y2-axis-vals, color...)
plt.bar(x_indexes + width, y3-axis-vals, color...)

# replace xticks with labels of x-axis
plt.xticks(ticks=x_indexes, labels=ages_x)
</code></pre>
3) Has some neat tricks using Counter
<pre><code>from collections import Counter
language_counter = Counter() # initialize counter
for row in csv_reader:
    language_counter.update(row['key].split(';'))

# get the 15 most common counters
print(language_counter.most_common(15))
</code></pre>
4) When using Horizontal Bar using `barh` one has to be careful about switching x-axis and y-axis
5) For downsorting data, `list.reverse()` function can be used. This is done in-place.

## Notes from 3rd chapter (pie charts)
1) [Matplotlib wedge properties]("https://matplotlib.org/api/_as_gen/matplotlib.patches.Wedge.html") link - like for digtionary of wedgeprops
2) Use pie-charts for less than 5 items
3) explodes can be used from a list to explode out a specific pie
4) can also add shadow = True and startangle = 90
5) format string to show percent is `autopct='%1.1f%%'`

## Notes from 4th chapter (stacked plots / area plots)
1) Stacked plots are area plots that can show total and distribution within each x-axis category
2) Labels can be positioned appropriately using, for e.g., `plt.legend(loc='upper left')`. 
   - `loc` can also be a tuple like `(0.07, 0.05)` - for positioning from bottom, left 

## Notes from 5th chapter (fill area on line plots)
1) use `plt.fill_between(x-axis, to_what)`
2) use `alpha=0.25` for 25% transparency
3) use `where()` for conditions, `interploate=True` and `label= ""` for better clarity

## Notes from 6th chapter (histograms)
1) `plt.hist(ages, bins=5)` plots data in 5 bins. Age ranges are auto-generated
2) `edgecolor` property is used to delineate the bins. ```plt.hist(ages, bins=5, edgecolor = 'black')```
3) For very large data-sets use `log=True`
4) For a median line - for instance - use `plt.axvline()`

## Notes from 7th chapter (scatter-plots)
1) `plt.scatter(x, y, s=100)` # s is the size of scatter dots
2)  A scatter plot more beautified is `plt.scatter(x, y, s=100, c='green', edgecolor='black', linewidth=1, alpha=0.75)`
3) Scatter plots can be coloured with `c = [list of colors]`, instead of `c='green'`
4) Scatter plots can be color-mapped using the argument `cmap='Greens'` ... for instance. [other colormaps are linked here](https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html).
5) Colour bar represents which variable means what. This is done by instantiating from the method `cbar=plt.colorbar()` and using the `set_label` parameter to show the label for the colour bar.
6) For very large dataset, which has outliers, it is better to `log` the x and y axis using the methods<code><pre>plt.xscale('log')
plt.yscale('log')</code></pre>

## Notes from 8th chapter (time-series)
1) By default `plt.plot_date()` gives markers instead of connecting by a line. To rectify put `plt.plot_date(dates, y, linestyle='solid)`
2) To change (rotate) the axis it is a method on `figures` and not on `plt` object.
   - get current figure it is `plt.gcf()` for get current figure
   - put autofmt_xdate() on `plt.gcf().autofmt_xdate()`
3) Datetime formatting - comes from `mpldates`
4) To set the date format, `date_format = mpl_dates.DateFormatter('%b, %d %Y')` is made, and added as:
<code>plt.gca().xaxis.set_major_formatter(date_format)</code>
5) If the dates are out of order, use pandas to set it to date_time and sort for the x-axis.

## Notes from 9th chapter (plotting live data in realtime)
1) `count` from `itertools` library just iteratively counts to the next value.
2) To animate 
   - Import `FuncAnimation` from `matplotlib.animation`
   - Call it using `FuncAnimation(plt.gcf(), animate, interval=1000)`
   - Simply calling the above keeps plotting brand new line, without overriding old line.
   - To prevent that one should put `plt.cla()` ... to clear the axis before plotting
   - Explore `FuncAnimation` for setups. `plt.cla() `can also be avoided.

# Notes from 10th chapter (sub-plots)
1) When `plt` is used without instantiating a class, it is called `staple`. When we import it, it has a current state, with axes, etc.
2) `Figure` is the container holding our plot (the window which pops out).
   - `plt.gcf()` is used to get the current figure.
3) `Axes` are the actual plots
   - `plt.gca()` is used to get the current axis.
4) Many people like to use the object-oriented way - using `subplots` method, as follows:
   - `fig, ax = plt.subplots()`
5) subplots take no of rows and columns. By default it is 1 x 1.
6) Instead of `plt`, use `ax`.
   - for `plt.title`, `plt.xlabel` and `plt.ylabel`;
      - use `plt.set_title`, `plt.set_xlabel`, `plt.set_ylabel`.
7) `fig, ax = plt.subplots(nrows=2, ncols=2)` ... gives a nested list of lists for ax having multiple sub-plots
8) unpack ax list using `fig, (ax1, ax2) = plt.subplots(2, 1)`
9) `X-Axis` labels and `Title`s need not be duplicated.
10) x-axis can be shared using `sharex=True` in `plt.subplots`
11) can be plotted on multiple figures by doing: 
<code><pre>fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()</pre></code>
12) figures can be saved using `savefig()` on those objects.


