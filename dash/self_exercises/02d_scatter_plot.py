import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
# import plotly.express as px
import plotly.graph_objects as go

app = dash.Dash(__name__)

df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')

# prepare data (x, y, mode, name, marker, text)
data = [dict(x=df[df.continent==i]['gdp per capita'],
             y=df[df.continent==i]['life expectancy'],
             type='scatter',
             mode='markers',
             name=i,
             text=df[df.continent==i]['country'],
             opacity=0.7,
             marker_size=15,
             marker_line_width=0.5,
             marker_line_color='white',
             marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
             }
            ) for i in df.continent.unique()]

# layout preparation (title, xaxis, legend, hovermode)
layout = dict(
                xaxis={'type': 'log', 'title': 'GDP Per Capita'},
                yaxis={'title': 'Life Expectancy'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )

# make the graph
app.layout = html.Div([
    dcc.Graph(
        id='life-exp-vs-gdp',
        figure=
            dict(data=data, 
                 layout=layout)
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)
