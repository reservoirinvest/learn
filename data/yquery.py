# Better program, which works better than yfinance
import datetime

import numpy as np
import pandas as pd
from forex_python.converter import CurrencyRates
from yahooquery import Ticker

# Set the symbols
symdict = dict()
symdict['CC3.SI'] = 'STARHUB'
symdict['Z74.SI'] = 'Singtel'
# symdict['B2F.SI'] = 'M1'
# symdict['B2F.SI'] = 'M1'


# symdict['600500.SS'] = 'SINOCHEM'
# symdict['PTR'] = 'PETROCHINA'
# symdict['RDS-A'] = 'Shell'
# symdict['2222.SR'] = 'Aramco'
# symdict['BP'] = 'BP'
# symdict['TTE'] = 'Total'
# symdict['CVX'] = 'Chevron'
# symdict['ROSN.ME'] = 'Rosneft'

# Get ticker objects
d_ticks = {det: Ticker(sym) for sym, det in symdict.items()}

# Instantiate the currency converter
c = CurrencyRates()

# Build Dataframes
# . local currency
df = pd.concat([t.all_financial_data() for _, t in d_ticks.items()]).reset_index()
df.insert(0, 'company', df.symbol.map(symdict))

# . usd dataframe

time_periods = pd.to_datetime(df.asOfDate).unique()

dict_rates = dict()

for t in time_periods:
    
    t1 = pd.to_datetime(t)
    try:
        dict_rates[t1] = c.get_rates('USD', pd.to_datetime(t))
    except Exception as e:
        
        # Second chance!
        try:
            # 2 days subtracted to remove any holidays!!
            dict_rates[t1] = c.get_rates('USD', pd.to_datetime(t) - datetime.timedelta(days=2))
        except Exception as e:
            print(e)

float_cols = df.columns[(df.dtypes.values == np.dtype('float64'))]

rates = df.asOfDate.map(dict_rates)
df1 = df.assign(usd_rate=[r[c] for c, r in zip(df.currencyCode, rates)])

# convert to financial figures to USD
df1[float_cols] = df1[float_cols].divide(df1.usd_rate, axis='index')
df1.currencyCode = 'USD'

df_usd = df1[df.columns] # rever back to df's columns

# create a Pandas Excel writer using XlsxWriter as the engine.
dfs = {'local': df, 'usd': df_usd}

writer = pd.ExcelWriter('./data/telecoms.xlsx', engine='xlsxwriter')

for sheetname, df in dfs.items():
    df.to_excel(writer, sheet_name=sheetname, index=False, freeze_panes=(1, 1))
    worksheet = writer.sheets[sheetname]  # pull worksheet object
    for idx, col in enumerate(df):  # loop through all columns
        series = df[col]
        max_len = max((
            series.astype(str).map(len).max(),  # len of largest item
            len(str(series.name))  # len of column name/header
            )) + 1  # adding a little extra space
        worksheet.set_column(idx, idx, max_len)  # set column width
writer.save()

print("Completed generating report!")
