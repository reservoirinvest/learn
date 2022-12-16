# Illustrates basic core components
import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)

# initiate core components
# . these are labels followed by the core component itself, assembled within a div

# .. dropdowns
dd_lbl = html.Label('Dropdown')
dd =  dcc.Dropdown(
                options=[
                    dict(label='New York City', value='NYC'),
                    dict(label=u'Montréal', value='MTL'),
                    dict(label='San Francisco', value='SF')
                ],
                value='MTL'
            )
            

# .. multi dropdowns
mdd_lbl = html.Label('Multi-Dropdown')
mdd = dcc.Dropdown(
                        options=[
                            dict(label='New York City', value='NYC'),
                            dict(label=u'Montréal', value='MTL'),
                            dict(label='San Francisco', value='SF')
                        ],
                        value=['MTL', 'SF'],
                        multi=True
                    )

# .. radios
radio_lbl = html.Label('Radio Items')
radio = dcc.RadioItems(
                        options=[
                            dict(label='New York City', value='NYC'),
                            dict(label=u'Montréal', value='MTL'),
                            dict(label='San Francisco', value='SF')
                        ],
                        value='MTL'                        
                    )

# .. checkboxes
chkbox_lbl = html.Label('Check boxes')
chkbox = dcc.Checklist(
                    options=[
                        dict(label='New York City', value='NYC'),
                        dict(label=u'Montréal', value='MTL'),
                        dict(label='San Francisco', value='SF')
                    ],
                    value=['MTL', 'NYC' ]
)

# .. text input
txtbox_lbl = html.Label('Text Input')
txtbox = dcc.Input(value='MTL', type='text')

# .. slider
slider_lbl = html.Label('Slider')
slider = dcc.Slider(
                min=0,
                max=9,
                marks= {i: f'Label {i}' if i==1 else str(i) for i in range(1, 6)},
                value=5
)

app.layout = html.Div(children=[dd_lbl, dd, html.Hr(), 
                                mdd_lbl, mdd, html.Hr(), 
                                radio_lbl, radio, html.Hr(), 
                                chkbox_lbl, chkbox, html.Hr(),
                                txtbox_lbl, txtbox, html.Hr(),
                                slider_lbl, slider, html.Hr()],
                      style={'columnCount': 2})

if __name__ == "__main__":
    app.run_server(debug=True)
