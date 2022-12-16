# Preamble

This repository has information on plotly, with a bit of dash.

* plotly keeps changing. For latest info, it is better to follow what is said in [Getting Started](https://plotly.com/python/getting-started/) official documentation
* [Kaleido](https://github.com/plotly/Kaleido) package has been made for rendering plotly, which can be installed using pip. This supercedes ~~OCRA~~
* Go through and try exercises in [`$Overview.ipynb`](https://github.com/reservoirinvest/learn/blob/master/plotly/$Overview.ipynb)

## First test

Check this code in JupyterLab...
```python
import pandas as pd
import plotly.express as px

import plotly.io as pio
pio.renderers.default = 'iframe'

df = pd.read_csv('http://bit.ly/drinksbycountry')
fig = px.bar(df[['continent', 'beer_servings', 'spirit_servings']].melt(id_vars='continent'), x='continent', y='value', color='variable', barmode='group')
pio.show(fig)
```

...if it works well you are all set!

## Plotly in Github

* Cleanup your notebook outputs - before commits - as they make the files heavy.
* Use `.show('jupyterlab')`, `.show('svg')`, or `.show('png')` to show the picture in Github... else it will fail in Github! 

# General Information
1. Beautiful notes given in [Plotly Fundamentals](https://plot.ly/python/plotly-fundamentals/) - esepecially on [Creating and Updating Figures](https://plot.ly/python/creating-and-updating-figures/) for a good overview.

2. Key-Value pairs for plotly's dictionaries can be found in the [Full Reference](https://plot.ly/python/reference/) page. The left menu is simply presented as:
   - Trace Types
   - Layout

## Steps to initiate plotly
* After ver 4, one can directly run plotly in Jupyter, without using offline mode.
* For prior versions:
    1. Set the offline mode
    2. Import high-level chart objects using <pre><code>import plotly.graph_objects as go</code></pre>

## For a bare-bones bar-chart
### Setup
1. Set a `trace`, which is used for painting graph objects on top of a `figure` using:<code><pre>
trace = go.Bar(x=["Jan", "Feb", "Mar"], 
        y=[10, 11, 14]])</pre></code>
2. Play around with `layout` - that provides properties of the figure's layout. It can be in nested key-value pairs, for eg:
<code><pre>
layout = {"title":"First quarter sales",
         "xaxis":{"title":"Months"},
         "yaxis":{"title":"Units"}}
</code></pre>
3. Put the `layout` and `trace` in a `figure` container ... using:<code><pre>
fig = go.Figure(data = data)
</pre></code>
4. Show the figure container using:
   - in Jupyter Notebook:~~<pre><code>iplot(fig)~~ `pio.show(fig)`</code></pre>
   - outside Jupyter <code><pre>fig.show()</pre></code>

### Manipulate
1. Colour change of bar is done at `trace` with `marker` dict/class in `go.Bar` object
2. data can be directly passed from key-value as:
   ```
   iplot({"data":data, "layout":layout})
   ```
3. Extra text can be added to hover by adding `text` to `trace` like:
```
   trace = go.Bar(x = ["January", "February", "March"],
               y = [10, 11, 14],
               marker = dict(color = ['rgba(255,0,0,1)', 'rgba(204,204,204,1)', 'rgba(204,204,204,1)']),
              text = ["Below target", "Above target", "Above target"])
```
4. Grouped bar charts are made by 
    - multiple `trace`s
    - having the same x-axis data in `go.Bar`
    - giving a `name` to each trace
    - putting them in a `data` list
    - putting `{"barmode":"group"}` key value in the layout

5. Stacked charts can be made by making the `{"barmode":"stack"}` key value in the layout

## For Histograms
1. Bare-bones histogram is simply: 
   <pre><code>trace = go.Histogram(x = age)
    data = [trace]
    pio.show({"data": data})</pre></code>
2. If we *normalize* it, we get a frequency distribution. This is done using `histnorm="probability"` in `go.Histogram()`.
      - `Normalization` means that the sume of the areas under the curve sums up to 1.
3. Histograms can be overlay-ed with opacity or stacked.
4. Bins are controlled using the `xbins` argument as a dictionary.
5. Cumulative histograms are controlled by using argument `cumulative = dict(enabled=True)`

## For distribution plots
1. Figure factory can be used with `import plotly.figure_factory as ff`
2. `ff.creat_distplot()` takes `hist_data` argument as a list. So dataframe values are to be converted to a list using `df.Salary.values.tolist()` function
3. Figure title can be added using `fig["Layout"].update(title="Salary Distribution")`
4. For creating normal distribution in figure factory, `curve_type = "normal"` is to be used in `ff.create_distplot()`
5. Histogram and Rug plot can be disabled using `show_hist = False` and `show_rug = False`

# Other References
1. Juan Koppler's [YouTube video](https://youtu.be/87jyeklhTH8)