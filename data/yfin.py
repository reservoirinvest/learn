# Program to generate PnL, BalanceSheet and Cashflow spreadsheets
# picked up from yfinance (yahoo finance)
# !! NOTE: yquery is a better one!

import yfinance as yf
import pandas as pd

def flatten_me(df: pd.DataFrame, 
                c: str, # symbol
                typ: str, # bs for balancesheet, cf for cashflow, pl for income
                period: str, # year or quarter
                ) -> pd.DataFrame:
    """
    flattens the yahoo financial extract and adds symbol and type
    """
    df1 = df.rename_axis('item').reset_index().melt(id_vars=['item'], 
                    value_vars=df.columns, var_name='date', value_name='usd')

    df1.insert(0, 'symbol', c)
    df1.insert(1, 'type', typ)
    df1.insert(2, 'period', period)

    return df1

co_list = ['WHR', 'ELUXY', 'HYHIY', 'HRLEY', 'BSCH']

result = []


for c in co_list:

    company = yf.Ticker(c)

    # balance sheet
    result.append(flatten_me(company.balance_sheet, c, 'bs', 'year'))
    result.append(flatten_me(company.quarterly_balance_sheet, c, 'bs', 'quarter'))

    # cash flow
    result.append(flatten_me(company.cashflow, c, 'cf', 'year'))
    result.append(flatten_me(company.quarterly_cashflow, c, 'cf', 'quarter'))

    # earnings
    result.append(flatten_me(company.financials, c, 'pl', 'year'))
    result.append(flatten_me(company.quarterly_financials, c, 'pl', 'quarter'))

df_fin = pd.concat(result, axis=0).reset_index(drop=True)
df_fin = df_fin.assign(date=pd.to_datetime(df_fin.date, format="%Y-%m-%d %H:%M:%S", errors='coerce'))

# create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('./data/finstat.xlsx', engine='xlsxwriter')

df_fin.to_excel(writer, sheet_name='finstat', float_format="%.0f", index=False, freeze_panes=(1, 1))

# master list of items and types
df_items = df_fin.groupby(['type', 'item']).size().reset_index(name='freq').drop(['freq'], 1)
df_items.to_excel(writer, sheet_name='type_item', index=False, freeze_panes=(1,1))

# close the Pandas Excel writer and output the Excel file.
writer.save()

print("Completed generating report!")