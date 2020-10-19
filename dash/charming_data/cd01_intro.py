import os
import pathlib
import sys

import dash  # version 1.12.1
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px  # version 4.11.0
import plotly.graph_objects as go
from dash.dependencies import Input, Output

# print(f"sys.path: {sys.path[0]}")
# print(pd.read_csv(sys.path[0]+r"\intro_bees.csv"))

print(f'os.path.dirname(__file__): {os.path.dirname(__file__)}\n')

print(f"pathlib.Path('.'): {pathlib.Path('.')}\n")

print(f'pathlib.Path.cwd(): {pathlib.Path.cwd()}\n')

print(f'pathlib.Path.home(): {pathlib.Path.home()}\n')


""" app = dash.Dash(__name__)

# --------------------------------------------
# Import and clean data (importing csv into pandas)
df = pd.read_csv("intro_bees.csv")

df = df.groupby(["State", "ANSI", "Affected by", "Year", "state_code"])[
    ["Pct of Colonies Impacted"]
].mean()
df.reset_index(inplace=True)
print(df.head()) """
