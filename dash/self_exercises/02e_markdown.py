# While Dash exposes HTML, using dash_html_components library, its tedious
# HTML Markup can be used instead

import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

markdown_text = '''
### Dash and Markdown

Dash apps can be written in Markdown.
Dash uses the [CommonMark](http://commonmark.org/)
specification of Markdown.
Checkout their [60 Second Markdown Tutorial](http://commonmark.org/help/)
if this is your first introduction to Markdown!
'''
app.layout = html.Div([
    dcc.Markdown(children=markdown_text)
])

if __name__ == "__main__":
    app.run_server(debug=True)
