# ** SETUP
# .Imports
import asyncio
import math
import pathlib
import time
from io import StringIO
from typing import Callable, Coroutine, Union

# .Specific to Jupyter. Will be ignored in IDE / command-lines
import IPython as ipy
import numpy as np
import pandas as pd
import requests
import yaml
from ib_insync import IB, Contract, MarketOrder, Option, util

from support import get_dte

if ipy.get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    import nest_asyncio
    nest_asyncio.apply()
    util.startLoop()
    pd.options.display.max_columns = None


# ** SINGLE FUNCTIONS
# .NSE df
async def get_nse() -> pd.DataFrame:
    '''Make nse symbols, qualify them and put into a dataframe'''

    url = "https://www1.nseindia.com/content/fo/fo_mktlots.csv"

    try:
        req = requests.get(url)
        if req.status_code == 404:
            print(f'\n{url} URL contents not correct. 404 error!!!')
        df_symlots = pd.read_csv(StringIO(req.text))
    except requests.ConnectionError as e:
        print(f'Connection Error {e}')
    except pd.errors.ParserError as e:
        print(f'Parser Error {e}')

    df_symlots = df_symlots[list(df_symlots)[1:5]]

    # strip whitespace from columns and make it lower case
    df_symlots.columns = df_symlots.columns.str.strip().str.lower()

    # strip all string contents of whitespaces
    df_symlots = df_symlots.applymap(
        lambda x: x.strip() if type(x) is str else x)

    # remove 'Symbol' row
    df_symlots = df_symlots[df_symlots.symbol != "Symbol"]

    # melt the expiries into rows
    df_symlots = df_symlots.melt(
        id_vars=["symbol"], var_name="expiryM", value_name="lot"
    ).dropna()

    # remove rows without lots
    df_symlots = df_symlots[~(df_symlots.lot == "")]

    # convert expiry to period
    df_symlots = df_symlots.assign(
        expiryM=pd.to_datetime(df_symlots.expiryM, format="%b-%y")
        .dt.to_period("M")
        .astype("str")
    )

    # convert lots to integers
    df_symlots = df_symlots.assign(
        lot=pd.to_numeric(df_symlots.lot, errors="coerce"))

    # convert & to %26
    df_symlots = df_symlots.assign(
        symbol=df_symlots.symbol.str.replace("&", "%26"))

    # convert symbols - friendly to IBKR
    df_symlots = df_symlots.assign(symbol=df_symlots.symbol.str.slice(0, 9))
    ntoi = {"M%26M": "MM", "M%26MFIN": "MMFIN",
            "L%26TFH": "LTFH", "NIFTY": "NIFTY50"}
    df_symlots.symbol = df_symlots.symbol.replace(ntoi)

    # differentiate between index and stock
    df_symlots.insert(
        1, "ctype", np.where(
            df_symlots.symbol.str.contains("NIFTY"), "IND", "STK")
    )

    df_symlots['exchange'] = 'NSE'
    df_symlots['currency'] = 'INR'
    df_symlots['contract'] = [Contract(symbol=symbol, secType=ctype, exchange=exchange, currency=currency)
                              for symbol, ctype, exchange, currency
                              in zip(df_symlots.symbol, df_symlots.ctype,
                                     df_symlots.exchange, df_symlots.currency)]

    return df_symlots


# .SNP df
async def get_snp(ib: IB) -> pd.DataFrame():
    '''Generate symlots for SNP 500 weeklies + those in portfolio as a DataFrame'''

    # Get the weeklies
    dls = "http://www.cboe.com/products/weeklys-options/available-weeklys"

    try:
        data = pd.read_html(dls)
    except Exception as e:
        print(f'Error: {e}')

    df_ix = pd.concat([data[i].loc[:, :0] for i in range(1, 3)], ignore_index=True)\
        .rename(columns={0: 'symbol'})
    df_ix = df_ix[df_ix.symbol.apply(len) <= 5]
    df_ix['ctype'] = 'Index'
    df_ix['exchange'] = 'CBOE'

    df_eq = data[4].iloc[:, :1].rename(columns={0: 'symbol'})
    df_eq = df_eq[df_eq.symbol.apply(len) <= 5]
    df_eq['ctype'] = 'Stock'
    df_eq['exchange'] = 'SMART'

    df_weeklies = pd.concat([df_ix, df_eq], ignore_index=True)
    df_weeklies = df_weeklies.assign(
        symbol=df_weeklies.symbol.str.replace('[^a-zA-Z]', ''))

    # Generate the snp 500s
    try:
        s500 = list(pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                                 header=0, match='Symbol')[0].loc[:, 'Symbol'])
    except Exception as e:
        print(f'Error: {e}')

    # without dot in symbol
    snp500 = [s.replace('.', '') if '.' in s else s for s in s500]

    # Keep only equity weeklies that are in S&P 500, and all indexes in the weeklies
    df_symlots = df_weeklies[((df_weeklies.ctype == 'Stock') & (df_weeklies.symbol.isin(snp500))) |
                             (df_weeklies.ctype == 'Index')].reset_index(drop=True)

    pf = ib.portfolio()

    more_syms = {p.contract.symbol for p in pf} - set(df_symlots.symbol)

    df_syms = pd.concat([pd.DataFrame({'symbol': list(more_syms), 'ctype': 'STK', 'exchange': 'SMART'}),
                         pd.DataFrame({'symbol': list(more_syms),
                                       'ctype': 'IND', 'exchange': 'SMART'}),
                         df_symlots], ignore_index=False)

    # Append other symlot fields
    df_syms = df_syms.assign(
        expiry=None, lot=100, currency='USD').drop_duplicates().reset_index(drop=True)

    return df_syms


# .qualify
async def qualify(ib: IB, c: Union[pd.Series, list, Contract]) -> pd.Series:
    if isinstance(c, list) | isinstance(c, pd.Series):
        result = await ib.qualifyContractsAsync(*c)
    elif isinstance(c, Contract):
        result = await ib.qualifyContractsAsync(c)
    else:
        result = None

    return pd.Series(result, name='contract')


# .OHLC
async def ohlc(ib: IB, c, DURATION: int = 365, OHLC_DELAY: int = 5) -> pd.DataFrame:
    ohlc = await ib.reqHistoricalDataAsync(
        contract=c,
        endDateTime="",
        durationStr=str(DURATION) + ' D',
        barSizeSetting="1 day",
        whatToShow="Trades",
        useRTH=True)
    await asyncio.sleep(OHLC_DELAY)
    df = util.df(ohlc)
    try:
        df.insert(0, 'symbol', c.symbol)
    except AttributeError:
        df = None
    return df


# .Underlying
async def und(ib: IB, c, FILL_DELAY) -> pd.DataFrame:

    tick = ib.reqMktData(c, '456, 104, 106, 100, 101, 165', snapshot=False)
    await asyncio.sleep(FILL_DELAY)

    ib.cancelMktData(c)

    try:
        undPrice = next(x for x in (tick.last, tick.close)
                        if not math.isnan(x))
    except Exception as e:
        print(f'undPrice not found in {tick.contract.symbol}. Error: {e}')
        undPrice = None

    m_df = pd.DataFrame(util.df([tick]))
    m_df['undPrice'] = undPrice

    div_df = pd.DataFrame(m_df.dividends.tolist())
    df1 = m_df.drop('dividends', 1).join(div_df)
    df1.insert(0, 'symbol', [c.symbol for c in df1.contract])
    df1['contract'] = c

    df2 = df1.dropna(axis=1)

    # Extract columns with legit values in them
    df3 = df2[[c for c in df2.columns if df2.loc[0, c]]]

    return df3


# .Chain
async def chain(ib: IB, c) -> pd.DataFrame:

    chains = await ib.reqSecDefOptParamsAsync(underlyingSymbol=c.symbol,
                                              futFopExchange="",
                                              underlyingSecType=c.secType,
                                              underlyingConId=c.conId)

    # Pick up one chain if it is a list
    chain = chains[0] if isinstance(chains, list) else chains

    df1 = pd.DataFrame([chain])

    # Do a cartesian merge
    df2 = pd.merge(pd.DataFrame(df1.expirations[0], columns=['expiry']).assign(key=1),
                   pd.DataFrame(df1.strikes[0], columns=['strike']).assign(key=1), on='key').\
        merge(df1.assign(key=1)).rename(columns={'tradingClass': 'symbol', 'multiplier': 'mult'})[
        ['symbol', 'expiry', 'strike', 'exchange', 'mult']]

    # Replace tradingclass to reflect correct symbol name of 9 characters
    df2 = df2.assign(symbol=df2.symbol.str[:9])

    # convert & to %26
    df2 = df2.assign(
        symbol=df2.symbol.str.replace("&", "%26"))

    # convert symbols - friendly to IBKR
    df2 = df2.assign(symbol=df2.symbol.str.slice(0, 9))
    ntoi = {"M%26M": "MM", "M%26MFIN": "MMFIN",
            "L%26TFH": "LTFH", "NIFTY": "NIFTY50"}
    df2.symbol = df2.symbol.replace(ntoi)

    # Get the dte
    df2['dte'] = df2.expiry.apply(get_dte)

    df2.loc[df2.dte <= 0, 'dte'] = 1  # Make dte = 1 for last day

    return df2


class Vars:
    """Variables from var.yml"""

    def __init__(self, MARKET: str) -> None:

        self.MARKET = MARKET

        with open('var.yml', 'rb') as f:
            data = yaml.safe_load(f)

        for k, v in data['COMMON'].items():
            setattr(self, k, v)

        for k, v in data[MARKET].items():
            setattr(self, k, v)


class Timer:
    """Timer providing elapsed time"""

    def __init__(self, name: str = '') -> None:
        self.name = name
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise Exception(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> None:
        if self._start_time is None:
            raise Exception(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"\n{self.name} time taken: " +
              f"{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} seconds\n")


if __name__ == "__main__":

    # * SET THE VARIABLES
    MARKET = 'SNP'

    ibp = Vars(MARKET.upper())  # IB Parameters from var.yml

    # * CLEAR THE LOGS
    LOGFILE = pathlib.Path.cwd().joinpath(
        'data', 'log', MARKET.lower() + '_base.log')

    util.logToFile(path=LOGFILE, level=30)

    with open(LOGFILE, 'w'):
        pass

    # * GET THE SYMLOTS

    symlots_time = Timer()
    symlots_time.start()

    with IB().connect(ibp.HOST, ibp.PORT, ibp.CID) as ib:
        df_symlots = ib.run(
            get_nse()) if MARKET.upper == 'NSE' else ib.run(get_snp(ib))

    print(df_symlots)
    symlots_time.stop()
