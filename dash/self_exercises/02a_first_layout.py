# "layout" describes what application looks like
# "interactivity" - covered in next chapter
# dcc = dash core components, dcc_html
# let us make an app.py

# Note that dash supports "hot-reloading" if data is changed in it
import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Hello to Dash from Kashi'),
    
    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    
    dcc.Graph(
        id='example-graph',
        figure={ # go figure is wrapped inside dcc.Graph
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},               
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )
])
