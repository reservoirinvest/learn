# ** SETUP
# .Imports
import asyncio
import math
import os
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
from pandas.core.algorithms import isin

from support import calcsd, calcsdmult_df, get_dte

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
VAR_YML = os.path.join(THIS_FOLDER, "var.yml")

if ipy.get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    import nest_asyncio

    nest_asyncio.apply()
    util.startLoop()
    pd.options.display.max_columns = None


# ** SINGLE FUNCTIONS
# .NSE df
async def get_nse() -> pd.DataFrame:
    """Make nse symbols, qualify them and put into a dataframe"""

    url = "https://www1.nseindia.com/content/fo/fo_mktlots.csv"

    try:
        req = requests.get(url)
        if req.status_code == 404:
            print(f"\n{url} URL contents not correct. 404 error!!!")
        df_symlots = pd.read_csv(StringIO(req.text))
    except requests.ConnectionError as e:
        print(f"Connection Error {e}")
    except pd.errors.ParserError as e:
        print(f"Parser Error {e}")

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
        1, "secType", np.where(
            df_symlots.symbol.str.contains("NIFTY"), "IND", "STK")
    )

    df_symlots["exchange"] = "NSE"
    df_symlots["currency"] = "INR"
    df_symlots["contract"] = [
        Contract(symbol=symbol, secType=secType,
                 exchange=exchange, currency=currency)
        for symbol, secType, exchange, currency in zip(
            df_symlots.symbol,
            df_symlots.secType,
            df_symlots.exchange,
            df_symlots.currency,
        )
    ]

    return df_symlots


# .SNP df
async def get_snp(ib: IB) -> pd.DataFrame():
    """Generate symlots for SNP 500 weeklies + those in portfolio as a DataFrame"""

    # Get the weeklies
    dls = "http://www.cboe.com/products/weeklys-options/available-weeklys"

    try:
        data = pd.read_html(dls)
    except Exception as e:
        print(f"Error: {e}")

    df_ix = pd.concat(
        [data[i].loc[:, :0] for i in range(1, 3)], ignore_index=True
    ).rename(columns={0: "symbol"})
    df_ix = df_ix[df_ix.symbol.apply(len) <= 5]
    df_ix["secType"] = "IDX"
    df_ix["exchange"] = "CBOE"

    df_eq = data[4].iloc[:, :1].rename(columns={0: "symbol"})
    df_eq = df_eq[df_eq.symbol.apply(len) <= 5]
    df_eq["secType"] = "STK"
    df_eq["exchange"] = "SMART"

    df_weeklies = pd.concat([df_ix, df_eq], ignore_index=True)
    df_weeklies = df_weeklies.assign(
        symbol=df_weeklies.symbol.str.replace("[^a-zA-Z]", "")
    )

    # Generate the snp 500s
    try:
        s500 = list(
            pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                header=0,
                match="Symbol",
            )[0].loc[:, "Symbol"]
        )
    except Exception as e:
        print(f"Error: {e}")

    # without dot in symbol
    snp500 = [s.replace(".", "") if "." in s else s for s in s500]

    # Keep only equity weeklies that are in S&P 500, and all indexes in the weeklies
    df_symlots = df_weeklies[
        ((df_weeklies.secType == "STK") & (df_weeklies.symbol.isin(snp500)))
        | (df_weeklies.secType == "IDX")
    ].reset_index(drop=True)

    pf = ib.portfolio()

    more_syms = {p.contract.symbol for p in pf} - set(df_symlots.symbol)

    df_syms = pd.concat(
        [
            pd.DataFrame(
                {"symbol": list(more_syms), "secType": "STK",
                 "exchange": "SMART"}
            ),
            pd.DataFrame(
                {"symbol": list(more_syms), "secType": "IND",
                 "exchange": "SMART"}
            ),
            df_symlots,
        ],
        ignore_index=False,
    )

    # Append other symlot fields
    df_syms = (
        df_syms.assign(expiry=None, lot=100, currency="USD")
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return df_syms


# .qualify
async def qualify(ib: IB, c: Union[pd.Series, list, tuple, Contract]) -> pd.Series:

    if isinstance(c, (list, pd.Series)):
        result = await ib.qualifyContractsAsync(*c)

    elif isinstance(c, tuple) & (c[1] is None):  # for contract qualification
        result = await ib.qualifyContractsAsync(c[0])

    elif isinstance(c, Contract):
        result = await ib.qualifyContractsAsync(c)

    else:
        result = None

    return pd.Series(result, name="contract")


# .OHLC
async def ohlc(ib: IB, c, DURATION: int = 365, OHLC_DELAY: int = 5) -> pd.DataFrame:

    if isinstance(c, tuple):
        c = c[0]

    ohlc = await ib.reqHistoricalDataAsync(
        contract=c,
        endDateTime="",
        durationStr=str(DURATION) + " D",
        barSizeSetting="1 day",
        whatToShow="Trades",
        useRTH=True,
    )
    await asyncio.sleep(OHLC_DELAY)
    df = util.df(ohlc)
    try:
        df.insert(0, "symbol", c.symbol)
    except AttributeError:
        df = None
    return df


# .Underlying
async def und(ib: IB, c, FILL_DELAY) -> pd.DataFrame:

    if isinstance(c, tuple):
        c = c[0]

    tick = ib.reqMktData(c, "456, 104, 106, 100, 101, 165", snapshot=False)
    await asyncio.sleep(FILL_DELAY)

    ib.cancelMktData(c)

    try:
        undPrice = next(x for x in (tick.last, tick.close)
                        if not math.isnan(x))
    except Exception as e:
        print(f"undPrice not found in {tick.contract.symbol}. Error: {e}")
        undPrice = None

    m_df = pd.DataFrame(util.df([tick]))
    m_df["undPrice"] = undPrice

    div_df = pd.DataFrame(m_df.dividends.tolist())
    df1 = m_df.drop("dividends", 1).join(div_df)
    df1.insert(0, "symbol", [c.symbol for c in df1.contract])
    df1["contract"] = c

    df2 = df1.dropna(axis=1)

    # Extract columns with legit values in them
    df3 = df2[[c for c in df2.columns if df2.loc[0, c]]]

    return df3


# .Chain
async def chain(ib: IB, c) -> pd.DataFrame:

    if isinstance(c, tuple):
        c = c[0]

    chains = await ib.reqSecDefOptParamsAsync(
        underlyingSymbol=c.symbol,
        futFopExchange="",
        underlyingSecType=c.secType,
        underlyingConId=c.conId,
    )

    # Pick up one chain if it is a list
    chain = chains[0] if isinstance(chains, list) else chains

    df1 = pd.DataFrame([chain])

    # Do a cartesian merge
    df2 = (
        pd.merge(
            pd.DataFrame(df1.expirations[0], columns=["expiry"]).assign(key=1),
            pd.DataFrame(df1.strikes[0], columns=["strike"]).assign(key=1),
            on="key",
        )
        .merge(df1.assign(key=1))
        .rename(columns={"tradingClass": "symbol", "multiplier": "mult"})[
            ["symbol", "expiry", "strike", "exchange", "mult"]
        ]
    )

    # Replace tradingclass to reflect correct symbol name of 9 characters
    df2 = df2.assign(symbol=df2.symbol.str[:9])

    # convert & to %26
    df2 = df2.assign(symbol=df2.symbol.str.replace("&", "%26"))

    # convert symbols - friendly to IBKR
    df2 = df2.assign(symbol=df2.symbol.str.slice(0, 9))
    ntoi = {"M%26M": "MM", "M%26MFIN": "MMFIN",
            "L%26TFH": "LTFH", "NIFTY": "NIFTY50"}
    df2.symbol = df2.symbol.replace(ntoi)

    # Get the dte
    df2["dte"] = df2.expiry.apply(get_dte)

    df2.loc[df2.dte <= 0, "dte"] = 1  # Make dte = 1 for last day

    return df2


# .Price
async def price(ib: IB, c) -> pd.DataFrame:

    if isinstance(c, tuple):
        c = c[0]

    tick = await ib.reqTickersAsync(c)

    try:
        dfpr = util.df(tick)
    except AttributeError:

        # return empty price df
        dfpr = pd.DataFrame({'localSymbol': {}, 'conId': {}, 'contract': {},
                             'undPrice': {}, 'bid': {}, 'ask': {}, 'close': {},
                             'last': {}, 'price': {}, 'iv': {}})

        return dfpr

    iv = dfpr.modelGreeks.apply(lambda x: x.impliedVol if x else None)
    undPrice = dfpr.modelGreeks.apply(lambda x: x.undPrice if x else None)
    price = dfpr['last'].combine_first(dfpr['close'])
    conId = dfpr.contract.apply(lambda x: x.conId if x else None)
    symbol = dfpr.contract.apply(lambda x: x.symbol if x else None)
    strike = dfpr.contract.apply(lambda x: x.strike if x else None)
    right = dfpr.contract.apply(lambda x: x.right if x else None)
    expiry = dfpr.contract.apply(
        lambda x: x.lastTradeDateOrContractMonth if x else None)

    dfpr = dfpr.assign(conId=conId, symbol=symbol, strike=strike, right=right, expiry=expiry, iv=iv,
                       undPrice=undPrice, price=price)

    cols = ['symbol', 'conId', 'strike', 'undPrice', 'expiry', 'right', 'contract', 'bid',
            'ask', 'close', 'last', 'price', 'iv']
    dfpr = dfpr[cols]

    return dfpr


# .Margin
async def margin(ib: IB, c) -> pd.DataFrame:

    ct, o = c

    async def wifAsync(ct, o):
        wif = ib.whatIfOrderAsync(ct, o)
        await asyncio.sleep(0)
        return wif

    if isinstance(c, tuple):
        wifs = await ib.whatIfOrderAsync(ct, o)

        df_wifs = pd.DataFrame([vars(wifs)])[['initMarginChange', 'maxCommission',
                                              'commission']]
        df_wifs = df_wifs.assign(conId=ct.conId, localSymbol=ct.localSymbol)
        df_wifs = df_wifs.assign(comm=df_wifs[['commission', 'maxCommission']].min(axis=1),
                                 margin=df_wifs.initMarginChange.astype('float'))
        df_margin = df_wifs[['conId', 'localSymbol', 'margin', 'comm']]
        df_margin = df_margin.assign(margin=np.where(df_margin.margin > 1e7, np.nan, df_margin.margin),
                                     comm=np.where(df_margin.comm > 1e7, np.nan, df_margin.comm))
    else:
        print(f"\nContract type of {c} is not tuple(c, o) !\n")
        df_margin = pd.DataFrame(
            [{'conId': ct.conId, 'localSymbol': ct.localSymbol, 'margin': np.nan, 'comm': np.nan}])

    return df_margin

# . Prepare raw options


def prepOpts(MARKET: str) -> list:
    """Prepare raw options for qualification"""

    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    ibp = Vars(MARKET)
    MAXDTE, MINDTE, CALLSTDMULT, PUTSTDMULT = ibp.MAXDTE, ibp.MINDTE, ibp.CALLSTDMULT, ibp.PUTSTDMULT

    # ... read the chains and unds
    df_symlots = pd.read_pickle(DATAPATH.joinpath('df_symlots.pkl'))
    df_chains = pd.read_pickle(DATAPATH.joinpath('df_chains.pkl'))
    df_unds = pd.read_pickle(DATAPATH.joinpath('df_unds.pkl'))

    # ... weed out chains within MIN and outside MAX dte
    df_ch = df_chains[df_chains.dte.between(MINDTE, MAXDTE, inclusive=True)
                      ].reset_index(drop=True)

    # ... get the undPrice and volatility
    df_u1 = df_unds[['symbol', 'undPrice', 'impliedVolatility']].rename(
        columns={'impliedVolatility': 'iv'})

    # ... integrate undPrice and volatility to df_ch
    df_ch = df_ch.set_index('symbol').join(
        df_u1.set_index('symbol')).reset_index()

    # ... get the expiries
    if MARKET == 'NSE':
        df_ch['expiryM'] = df_ch.expiry.apply(lambda d: d[:4] + '-' + d[4:6])
        cols1 = ['symbol', 'expiryM']
        df_ch = df_ch.set_index(cols1).join(
            df_symlots[cols1 + ['lot']].set_index(cols1)).reset_index()
        df_ch = df_ch.drop('expiryM', 1)
    else:
        df_ch['lot'] = 100

    # ...compute one standard deviation and mask out chains witin the fence
    df_ch = df_ch.assign(sd1=df_ch.undPrice * df_ch.iv *
                         (df_ch.dte / 365).apply(math.sqrt))

    fence_mask = (df_ch.strike > df_ch.undPrice + df_ch.sd1 * CALLSTDMULT) | \
        (df_ch.strike < df_ch.undPrice - df_ch.sd1 * PUTSTDMULT)

    df_ch1 = df_ch[fence_mask].reset_index(drop=True)

    # ... identify puts and the calls
    df_ch1.insert(3, 'right', np.where(
        df_ch1.strike < df_ch1.undPrice, 'P', 'C'))

    # ... remove duplicates, if any
    df_ch1 = df_ch1.drop_duplicates(
        subset=['symbol', 'expiry', 'strike', 'right'])

    # ... build the options
    opts = [Option(s, e, k, r, x)
            for s, e, k, r, x
            in zip(df_ch1.symbol, df_ch1.expiry, df_ch1.strike,
                   df_ch1.right, ['NSE' if MARKET.upper() == 'NSE' else 'SMART'] * len(df_ch1))]

    return opts


# .Execution
def remains(cts):
    """Generates tuples for tracking remaining contracts"""

    if isinstance(cts, Contract):  # Single contract
        remaining = cts, None  # Convert to tuple with None

    elif isinstance(cts, pd.Series):  # Contracts given as a series
        if len(cts) == 1:  # Single contract given as a series
            remaining = list(cts)[0], None  # Convert to a tuple with None
        else:
            remaining = tuple((c, None) for c in cts)

    elif isinstance(cts, list):  # List of Contracts or (c, o) tuples
        if len(cts) == 1:  # Single contract or (c, o)
            if isinstance(cts[0], tuple):  # (c, o) tuple
                remaining = tuple(cts[0])
            else:  # Single contract
                try:
                    remaining = cts[0].iloc[0], None
                except AttributeError:
                    remaining = cts[0], None
        else:  # Multiple contracts or (c, o)
            if isinstance(cts[0], tuple):  # (c, o) tuples
                remaining = tuple((c, o) for c, o in cts)
            else:  # Multiple contracts
                remaining = tuple((c, None) for c in cts)

    elif isinstance(cts, tuple):
        if len(cts) == 2:  # Single (c, o) tuple
            remaining = cts

    else:
        remaining = None

    return remaining


async def executeAsync(
    ib: IB(),
    algo: Callable[..., Coroutine],  # coro name
    cts: Union[Contract, pd.Series, list, tuple],  # list of contracts
    post_process: Callable[[set, pathlib.Path, str],
                           pd.DataFrame] = None,  # If checkpoint is needed
    DATAPATH: pathlib.Path = None,  # Necessary for post_process
    CONCURRENT: int = 40,  # adjust to prevent overflows
    TIMEOUT: None = None,  # if None, no progress messages shown
    OP_FILENAME: str = "",  # output file name
    **kwargs,  # keyword inputs for algo
):

    tasks = set()
    results = set()
    remaining = remains(cts)
    last_len_tasks = 0  # tracking last length for error catch

    # Determine unique names for tasks
    ct_name = "c[0].symbol+c[0].lastTradeDateOrContractMonth[-4:]+c[0].right+str(c[0].strike)+'..'"

    # Get the results
    while len(remaining):

        # Tasks limited by concurrency
        if len(remaining) <= CONCURRENT:

            # For single contract
            if (len(remaining) == 2) & (isinstance(remaining[1], (type(None), MarketOrder))):
                tasks.update([asyncio.create_task(
                    algo(ib, remaining, **kwargs), name=remaining[0].localSymbol)])

            else:
                tasks.update(
                    asyncio.create_task(
                        algo(ib, c, **kwargs), name=eval(ct_name))
                    for c in remaining
                )
        else:
            tasks.update(
                asyncio.create_task(algo(ib, c, **kwargs), name=eval(ct_name))
                for c in remaining[:CONCURRENT]
            )

        # Execute tasks
        while len(tasks):

            done, tasks = await asyncio.wait(
                tasks, timeout=TIMEOUT, return_when=asyncio.ALL_COMPLETED
            )

            # Remove dones from remaining
            done_names = [d.get_name() for d in done]
            try:
                remaining = [c for c in remaining if eval(
                    ct_name) not in done_names]
            except TypeError:  # completed the single contract!
                remaining = []

            # Update results and checkpoint
            results.update(done)

            # Checkpoint the results
            if post_process:
                output = post_process(results, DATAPATH, OP_FILENAME)
            else:
                output = results

            if TIMEOUT:
                if not isinstance(cts, Contract):  # len(cts) fails for a single contract!
                    print(
                        f"\nDone {algo.__name__} for {done_names[:2]} {len(results)} out of {len(cts)}. Pending {[eval(ct_name) for c in remaining][:2]}"
                    )

                # something wrong. Task is not progressing
                if (len(tasks) == last_len_tasks) & (len(tasks) > 0):
                    print(
                        f"\n @ ALERT @: Tasks are not progressing. Pending tasks will be killed in 5 seconds !\n")
                    dn, pend = await asyncio.wait(tasks, timeout=5.0)
                    if len(dn) > 0:
                        results.update(dn)

                    tasks.difference_update(dn)
                    tasks.difference_update(pend)

                    pend_names = [p.get_name() for p in pend]
                    try:
                        # remove pending from remaining
                        remaining = [c for c in remaining if eval(
                            ct_name) not in pend_names]

                    except TypeError:  # completed single contract!
                        remaining = []

                # re-initialize last length of tasks
                last_len_tasks = len(tasks)

    return output


def save_df(results: set, DATAPATH: pathlib.Path, file_name: str = "") -> pd.DataFrame():

    if results:
        df = pd.concat([r.result() for r in results if r], ignore_index=True)
        if file_name:
            df.to_pickle(DATAPATH.joinpath(file_name))
    else:
        df = pd.DataFrame([])  # results are not yet ready!
    return df


class Vars:
    """Variables from var.yml"""

    def __init__(self, MARKET: str) -> None:

        self.MARKET = MARKET

        with open(VAR_YML, "rb") as f:
            data = yaml.safe_load(f)

        for k, v in data["COMMON"].items():
            setattr(self, k, v)

        for k, v in data[MARKET].items():
            setattr(self, k, v)


class Timer:
    """Timer providing elapsed time"""

    def __init__(self, name: str = "") -> None:
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
        print(
            f"\n...{self.name} took: "
            + f"{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} seconds\n"
        )


if __name__ == "__main__":

    # * SET THE VARIABLES
    MARKET = "SNP"
    ibp = Vars(MARKET.upper())  # IB Parameters from var.yml

    HOST, PORT, CID = ibp.HOST, ibp.PORT, ibp.CID

    LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")
    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    # * SETUP LOGS AND CLEAR THEM
    LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_base.log")
    util.logToFile(path=LOGFILE, level=30)
    with open(LOGFILE, "w"):
        pass

    # .. start the timer
    all_time = Timer("Complete program")
    all_time.start()

    # * GET THE SYMLOTS
    with IB().connect(ibp.HOST, ibp.PORT, ibp.CID) as ib:
        df_symlots = (
            ib.run(get_nse()) if MARKET.upper(
            ) == "NSE" else ib.run(get_snp(ib))
        )

        # to prevent first TimeoutError()
        ib.disconnect()
        IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

    # ... prepare underlying contracts
    df_sl = df_symlots.drop_duplicates(subset=["symbol"])

    raw_unds = [
        Contract(secType=st, symbol=sym, exchange=exch, currency=curr)
        for st, sym, exch, curr in zip(
            df_sl.secType,
            df_sl.symbol,
            df_sl.exchange,
            df_sl.currency,
        )
    ]

    # * QUALIFY THE UNDERLYINGS

    u_qual_time = Timer(MARKET.lower() + "_und_qual")
    u_qual_time.start()

    with IB().connect(HOST, PORT, CID) as ib:
        und_cts = ib.run(qualify(ib, raw_unds))
        df_symlots = df_symlots.assign(
            contract=df_symlots.symbol.map({u.symbol: u for u in und_cts})
        )
        df_symlots.to_pickle(DATAPATH.joinpath("df_symlots.pkl"))

        # to prevent first TimeoutError()
        ib.disconnect()
        IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

    u_qual_time.stop()

    # ... list of underlying contracts
    df_symlots = pd.read_pickle(DATAPATH.joinpath("df_symlots.pkl"))
    und_cts = list(df_symlots.contract.unique())
    # und_cts = und_cts[8]  # !!! DATA LIMITER

    # * GET OHLCS
    ohlc_time = Timer(MARKET.lower() + "_ohlcs")
    ohlc_time.start()

    with IB().connect(HOST, PORT, CID) as ib:
        df_ohlcs = ib.run(executeAsync(ib=ib, algo=ohlc, cts=und_cts,
                                       CONCURRENT=20, TIMEOUT=15,
                                       post_process=save_df, DATAPATH=DATAPATH, OP_FILENAME='df_ohlcs.pkl',
                                       **{'DURATION': 365, 'OHLC_DELAY': 20},
                                       ))
    ohlc_time.stop()

    # * GET UNDERLYINGS
    und_time = Timer(MARKET.lower() + "_unds")
    und_time.start()

    with IB().connect(HOST, PORT, CID) as ib:
        df_unds = ib.run(executeAsync(ib=ib, algo=und, cts=und_cts,
                                      CONCURRENT=40, TIMEOUT=5,
                                      post_process=save_df, DATAPATH=DATAPATH, OP_FILENAME='df_unds.pkl',
                                      **{'FILL_DELAY': 8},
                                      ))
    und_time.stop()

    # * GET CHAINS
    chain_time = Timer(MARKET.lower() + "_chains")
    chain_time.start()

    with IB().connect(HOST, PORT, CID) as ib:
        df_chains = ib.run(executeAsync(ib=ib, algo=chain, cts=und_cts,
                                        CONCURRENT=44, TIMEOUT=5,
                                        post_process=save_df, DATAPATH=DATAPATH, OP_FILENAME='df_chains.pkl',
                                        ))
    chain_time.stop()

    # * GET UNDERLYING PRICES
    und_pr_time = Timer(MARKET.lower() + "_und_prices")
    und_pr_time.start()

    with IB().connect(HOST, PORT, CID) as ib:
        df_und_price = ib.run(executeAsync(ib=ib, algo=price, cts=und_cts,
                                           CONCURRENT=50, TIMEOUT=18,
                                           post_process=save_df, DATAPATH=DATAPATH, OP_FILENAME='df_und_prices.pkl',
                                           ))
    und_pr_time.stop()

    # * GET UNDERLYING MARGINS
    und_mgn_time = Timer(MARKET.lower() + "und_margins")
    und_mgn_time.start()

    df_syms = df_symlots.drop_duplicates(subset=['symbol'])
    und_cts = df_syms.contract.to_list()
    und_ords = [MarketOrder('SELL', 100)] * len(df_syms) \
        if MARKET.upper() == 'SNP' \
        else [MarketOrder('SELL', q) for q in df_syms.lot]

    und_cos = [(c, o) for c, o in zip(und_cts, und_ords)]

    with IB().connect(HOST, PORT, CID) as ib:
        df_und_margins = ib.run(executeAsync(ib=ib, algo=margin, cts=und_cos,
                                             CONCURRENT=200, TIMEOUT=5,
                                             post_process=save_df, DATAPATH=DATAPATH, OP_FILENAME='df_und_margins.pkl'))

    und_mgn_time.stop()

    # * PREPARE RAW OPTIONS AND QUALIFY THEM
    qopt_time = Timer('qualify_opts')
    qopt_time.start()

    raw_opts = prepOpts(MARKET)

    with IB().connect(HOST, PORT, CID) as ib:
        qopts = ib.run(executeAsync(ib=ib, algo=qualify, cts=raw_opts,
                                    CONCURRENT=200, TIMEOUT=5,
                                    post_process=save_df, DATAPATH=DATAPATH, OP_FILENAME='qopts.pkl'))
    qopt_time.stop()

    # * GET PRICE OF THE QUALIFIED OPTIONS
    qopts = pd.read_pickle(DATAPATH.joinpath('qopts.pkl'))

    qopt_price_time = Timer('qopt_prices')
    qopt_price_time.start()

    with IB().connect(HOST, PORT, CID) as ib:
        df_opt_prices = ib.run(executeAsync(ib=ib, algo=price, cts=qopts,
                                            CONCURRENT=100, TIMEOUT=8,
                                            post_process=save_df, DATAPATH=DATAPATH, OP_FILENAME='df_opt_prices.pkl',
                                            ))

    qopt_price_time.stop()

    # * GET MARGINS OF THE QUALIFIED OPTIONS
    opt_mgn_time = Timer(MARKET.lower() + "opt_margins")
    opt_mgn_time.start()

    opt_cts = qopts

    optcols = "conId,symbol,lastTradeDateOrContractMonth,strike,right,exchange".split(
        ',')
    df_opts = util.df(qopts.to_list())[optcols].rename(
        columns={'lastTradeDateOrContractMonth': 'expiry'})
    df_opts['expiryM'] = df_opts.expiry.apply(lambda d: d[:4] + '-' + d[4:6])
    df_opts['contract'] = qopts

    # ...get the lots
    if MARKET.upper() == 'NSE':
        cols1 = ['symbol', 'expiryM']
        df_opts = df_opts.set_index(cols1).join(
            df_symlots[cols1 + ['lot']].set_index(cols1)).reset_index()
        df_opts = df_opts.drop('expiryM', 1)
    else:
        df_opts['lot'] = 100

    df_opts = df_opts.assign(order=[MarketOrder('SELL', lot / lot) if MARKET.upper() ==
                                    'SNP' else MarketOrder('SELL', lot) for lot in df_opts.lot])

    opt_cos = [(c, o) for c, o in zip(df_opts.contract, df_opts.order)]

    with IB().connect(HOST, PORT, CID) as ib:
        df_opt_margins = ib.run(executeAsync(ib=ib, algo=margin, cts=opt_cos,
                                             CONCURRENT=200, TIMEOUT=5,
                                             post_process=save_df, DATAPATH=DATAPATH, OP_FILENAME='df_opt_margins.pkl'))
    opt_mgn_time.stop()

    # * INTEGRATE AND PICKLE FINAL OPTION df_opts
    # !!! TEMPORARY for integration test
    # df_symlots = pd.read_pickle(DATAPATH.joinpath('df_symlots.pkl'))
    # df_unds = pd.read_pickle(DATAPATH.joinpath('df_unds.pkl'))
    # df_opt_margins = pd.read_pickle(DATAPATH.joinpath('df_opt_margins.pkl'))
    # df_opt_prices = pd.read_pickle(DATAPATH.joinpath('df_opt_prices.pkl'))

    # ... replace missing undPrice with price from df_und
    old_price = df_opt_prices[df_opt_prices.undPrice.isnull()].symbol.map(
        df_unds.set_index('symbol').undPrice.to_dict())
    df_opt_prices.loc[df_opt_prices.undPrice.isnull(),
                      'undPrice'] = old_price

    # ... merge price, margins and lots
    cols_to_use = list(df_opt_margins.columns.difference(
        df_opt_prices.columns)) + ['conId']
    df = df_opt_prices.set_index('conId').join(
        df_opt_margins[cols_to_use].set_index('conId')).reset_index()

    # ... add intrinsic and time values
    df = df.assign(intrinsic=np.where(df.right == 'C',
                                      (df.undPrice - df.strike).clip(0, None),
                                      (df.strike - df.undPrice).clip(0, None)))
    df = df.assign(timevalue=df.price - df.intrinsic,
                   dte=df.expiry.apply(get_dte))

    # ... compute std deviation
    df.insert(14, 'sdMult', calcsdmult_df(df.strike, df))

    # ... get the lots
    df['expiryM'] = df.expiry.apply(lambda d: d[:4] + '-' + d[4:6])
    if MARKET.upper() == 'NSE':
        cols1 = ['symbol', 'expiryM']
        df = df.set_index(cols1).join(
            df_symlots[cols1 + ['lot']].set_index(cols1)).reset_index()
        df = df.drop('expiryM', 1)
    else:
        df['lot'] = 100

    # ... compute rom
    df['rom'] = (df.price * df.lot - df.comm).clip(0) / \
        df.margin * 365 / df.dte

    # ... get the underlying iv and std for reference
    und_iv = df.symbol.map({i: j for k, v in df_unds[['symbol', 'impliedVolatility']].
                            set_index('symbol').to_dict().items() for i, j in v.items()})

    und_sd = df[['symbol', 'strike', 'undPrice', 'dte']].\
        assign(iv=und_iv).\
        apply(lambda x: calcsd(x.strike, x.undPrice, x.dte, x.iv), axis=1)

    df = df.assign(undIV=und_iv, undSD=und_sd)

    # ... pickle
    df.to_pickle(DATAPATH.joinpath('df_opts.pkl'))

    all_time.stop()
