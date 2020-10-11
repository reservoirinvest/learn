# Testing how this works outside Jupyter!
import plotly.graph_objs as go

# Set the trace
trace = go.Bar(x=["Jan", "Feb", "Mar"],
               y=[10, 11, 14])

# build the data list
data = [trace]

# Set the layout
layout = {"title": "First quarter sales",
          "xaxis": {"title": "Months"},
          "yaxis": {"title": "Units"}}

# Putting in to the figure container
fig = go.Figure(data=data, layout=layout)

# Show the figure!
fig.show()
