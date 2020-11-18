# ** SETUP
# .Imports
import asyncio
import math
import os
import pathlib
from io import StringIO
from typing import Callable, Coroutine, Union

import IPython as ipy
import numpy as np
import pandas as pd
import requests
from ib_insync import IB, Contract, MarketOrder, Option, util

from support import Timer, Vars, get_dte

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

# ** SINGLE FUNCTIONS
# * Independent functions - ready to be called.
# .NSE df


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

    return pd.Series(result, name="contract", dtype=object)


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
async def und(ib: IB, c, FILL_DELAY=8) -> pd.DataFrame:
    """Use CONCURRENT=40, TMEOUT=8 for optimal executeAsync results"""

    if isinstance(c, tuple):
        c = c[0]

    tick = ib.reqMktData(c, "456, 104, 106, 100, 101, 165", snapshot=False)
    await asyncio.sleep(FILL_DELAY)

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
    df1.insert(1, "secType", [c.secType for c in df1.contract])

    df2 = df1.dropna(axis=1)

    # Extract columns with legit values in them
    df3 = df2[[c for c in df2.columns if df2.loc[0, c]]].reset_index(drop=True)

    # . Implied volatility correction

    # rename column
    df3.rename(columns={"impliedVolatility": "iv"}, inplace=True)

    ib.cancelMktData(c)

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

    try:
        chain = chains[0] if isinstance(chains, list) else chains
    except IndexError as ie:
        print(f"\nThere is something wrong in chain for {c.symbol}: {ie}\n")
        df = pd.DataFrame([])
        return df  # ! Abort chain generation

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
async def price(ib: IB, c, FILL_DELAY=8) -> pd.DataFrame:
    """Price coro. Use CONCURRENT=40, TMEOUT=8 for optimal executeAsync results"""

    if isinstance(c, tuple):
        c = c[0]

    tick = ib.reqMktData(c, genericTickList="106")

    await asyncio.sleep(FILL_DELAY)

    try:
        dfpr = util.df([tick])
        dfpr["symbol"] = c.symbol
        dfpr["secType"] = c.secType
        dfpr["localSymbol"] = c.localSymbol
        dfpr["conId"] = c.conId
        dfpr["contract"] = c
        dfpr["strike"] = c.strike
        dfpr["expiry"] = c.lastTradeDateOrContractMonth
        dfpr["right"] = c.right
        dfpr["bid"] = dfpr.bid
        dfpr["ask"] = dfpr.ask
        dfpr["close"] = dfpr.close
        dfpr["last"] = dfpr["last"]
        dfpr["price"] = dfpr["last"].combine_first(dfpr["close"])
        dfpr["time"] = dfpr.time
        dfpr["model"] = dfpr.modelGreeks
        if dfpr.model[0] is None:
            dfpr["iv"] = dfpr.impliedVolatility
        else:
            dfpr["iv"] = dfpr.model[0].impliedVol

    except AttributeError as e:

        print(f"\nError in {c.localSymbol}: {e}. df will be empty!\n")

        # return empty price df
        dfpr = pd.DataFrame([])  # initialize empty df

        dfpr["symbol"] = c.symbol
        dfpr["secType"] = c.secType
        dfpr["localSymbol"] = c.localSymbol
        dfpr["conId"] = c.conId
        dfpr["contract"] = c
        dfpr["strike"] = c.strike
        dfpr["expiry"] = c.lastTradeDateOrContractMonth
        dfpr["right"] = c.right
        dfpr["bid"] = np.nan
        dfpr["ask"] = np.nan
        dfpr["close"] = np.nan
        dfpr["last"] = np.nan
        dfpr["price"] = np.nan
        dfpr["iv"] = np.nan
        dfpr["time"] = np.nan
        dfpr["model"] = None

    cols = [
        "symbol",
        "secType",
        "localSymbol",
        "conId",
        "strike",
        "expiry",
        "right",
        "contract",
        "time",
        "model",
        "bid",
        "ask",
        "close",
        "last",
        "price",
        "iv",
    ]

    ib.cancelMktData(c)

    return dfpr[cols]


# .Margin
async def margin(ib: IB, c) -> pd.DataFrame:

    ct, o = c

    async def wifAsync(ct, o):
        wif = ib.whatIfOrderAsync(ct, o)
        await asyncio.sleep(0)
        return wif

    wifs = await ib.whatIfOrderAsync(ct, o)

    try:
        df_wifs = pd.DataFrame([vars(wifs)])[
            ["initMarginChange", "maxCommission", "commission"]
        ]
        df_wifs = df_wifs.assign(conId=ct.conId, localSymbol=ct.localSymbol)
        df_wifs = df_wifs.assign(
            comm=df_wifs[["commission", "maxCommission"]].min(axis=1),
            margin=df_wifs.initMarginChange.astype("float"),
        )
        df_margin = df_wifs[["conId", "localSymbol", "margin", "comm"]]
        df_margin = df_margin.assign(
            margin=np.where(df_margin.margin > 1e7, np.nan, df_margin.margin),
            comm=np.where(df_margin.comm > 1e7, np.nan, df_margin.comm),
        )
    except TypeError as e:
        print(
            f"\nTypeError: Something is wrong with contract: {c}, order: {0} !\n")
        df_margin = pd.DataFrame(
            [
                {
                    "conId": ct.conId,
                    "localSymbol": ct.localSymbol,
                    "margin": np.nan,
                    "comm": np.nan,
                }
            ]
        )

    df_margin.insert(2, "secType", ct.secType)

    return df_margin


# ** EXECUTION
# * Core engine that processes functions and delivers results

# .preprocessing data for the core engine
def pre_process(cts):
    """Generates tuples for input to the engine"""

    try:
        symbol = cts.symbol
        output = ((cts, None),)

    except AttributeError as ae1:  # it's an iterable!
        try:
            symbols = [c.symbol for c in cts]

            if len(symbols) == 1:
                output = ((cts[0], None),)
            else:
                output = ((c, None) for c in cts)

        except AttributeError as ae2:  # 2nd value is MarketOrder!
            try:
                output = tuple(cts)
            except:
                print(f"Unknown error in {ae2}")
                output = None

    return tuple(output)


# .make name for symbol being processed by the engine
def make_name(cts):
    """Generates name for contract(s)"""
    try:
        output = [
            c.symbol
            + c.lastTradeDateOrContractMonth[-4:]
            + c.right
            + str(c.strike)
            + ".."
            for c in cts
        ]

    except TypeError as te:  # single non-iterable element
        if cts:  # not empty!
            output = (
                cts.symbol
                + cts.lastTradeDateOrContractMonth[-4:]
                + cts.right
                + str(cts.strike)
            )
        else:
            output = cts

    except AttributeError as ae1:  # multiple (p, s) combination
        try:
            output = [
                c[0].symbol
                + c[0].lastTradeDateOrContractMonth[-4:]
                + c[0].right
                + str(c[0].strike)
                + ".."
                for c in cts
            ]
        except TypeError as ae2:
            output = (
                cts[0].symbol
                + cts[0].lastTradeDateOrContractMonth[-4:]
                + cts[0].right
                + str(cts[0].strike)
            )

    return output


# .the core engine
async def executeAsync(
    ib: IB(),
    algo: Callable[..., Coroutine],  # coro name
    cts: Union[Contract, pd.Series, list, tuple],  # list of contracts
    post_process: Callable[
        [set, pathlib.Path, str], pd.DataFrame
    ] = None,  # If checkpoint is needed
    DATAPATH: pathlib.Path = None,  # Necessary for post_process
    CONCURRENT: int = 40,  # adjust to prevent overflows
    TIMEOUT: None = None,  # if None, no progress messages shown
    OP_FILENAME: str = "",  # output file name
    **kwargs,  # keyword inputs for algo
):

    tasks = set()
    results = set()
    remaining = pre_process(cts)
    last_len_tasks = 0  # tracking last length for error catch

    # Get the results
    while len(remaining):

        # Tasks limited by concurrency
        if len(remaining) <= CONCURRENT:

            tasks.update(
                asyncio.create_task(algo(ib, c, **kwargs), name=make_name(c))
                for c in remaining
            )

        else:

            tasks.update(
                asyncio.create_task(algo(ib, c, **kwargs), name=make_name(c))
                for c in remaining[:CONCURRENT]
            )

        # Execute tasks
        while len(tasks):

            done, tasks = await asyncio.wait(
                tasks, timeout=TIMEOUT, return_when=asyncio.ALL_COMPLETED
            )

            # Remove dones from remaining
            done_names = [d.get_name() for d in done]
            remaining = [c for c in remaining if make_name(
                c) not in done_names]

            # Update results and checkpoint
            results.update(done)

            # Checkpoint the results
            if post_process:
                output = post_process(results, DATAPATH, OP_FILENAME)
            else:
                output = results

            if TIMEOUT:
                if remaining:
                    print(
                        f"\nDone {algo.__name__} for {done_names[:2]} {len(results)} out of {len(cts)}. Pending {[make_name(c) for c in remaining][:2]}"
                    )

                # something wrong. Task is not progressing
                if (len(tasks) == last_len_tasks) & (len(tasks) > 0):
                    print(
                        f"\n @ ALERT @: Tasks are not progressing. Pending tasks will be killed in 5 seconds !\n"
                    )
                    dn, pend = await asyncio.wait(tasks, timeout=5.0)
                    if len(dn) > 0:
                        results.update(dn)

                    tasks.difference_update(dn)
                    tasks.difference_update(pend)

                    pend_names = [p.get_name() for p in pend]
                    # remove pending from remaining
                    remaining = [c for c in remaining if make_name(
                        c) not in pend_names]

                # re-initialize last length of tasks
                last_len_tasks = len(tasks)

    return output


# .Process output into dataframes
def save_df(
    results: set, DATAPATH: pathlib.Path, file_name: str = ""
) -> pd.DataFrame():

    if results:
        df = pd.concat([r.result() for r in results if r], ignore_index=True)
        if file_name:
            df.to_pickle(DATAPATH.joinpath(file_name))
    else:
        df = pd.DataFrame([])  # results are not yet ready!
    return df


# ** ACTIVITIES
# * Function calls to achieve specific results


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
    df_ix["secType"] = "IND"
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
        | (df_weeklies.secType == "IND")
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


# .. generate symbols and lots
def get_symlots(MARKET: str, RUN_ON_PAPER: bool = False) -> pd.DataFrame:

    u_qual_time = Timer(MARKET.lower() + "_und_qual")
    u_qual_time.start()

    # ... set parameters from var.yml

    ibp = Vars(MARKET.upper())
    HOST, CID = ibp.HOST, ibp.CID
    if RUN_ON_PAPER:
        PORT = ibp.PAPER
    else:
        PORT = ibp.PORT

    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    # * GET THE SYMLOTS
    with IB().connect(HOST, PORT, CID) as ib:
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
            df_sl.secType, df_sl.symbol, df_sl.exchange, df_sl.currency,
        )
    ]

    with IB().connect(HOST, PORT, CID) as ib:
        und_cts = ib.run(
            executeAsync(
                ib=ib,
                algo=qualify,
                cts=raw_unds,
                CONCURRENT=100,
                TIMEOUT=5,
                post_process=save_df,
            )
        )

        df_symlots = df_symlots.assign(
            contract=df_symlots.symbol.map({u.symbol: u for u in und_cts})
        )

        # remove any NaN
        df_symlots = df_symlots.dropna(subset=["contract"])

        df_symlots.to_pickle(DATAPATH.joinpath("df_symlots.pkl"))

        # to prevent first TimeoutError()
        ib.disconnect()
        IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

        u_qual_time.stop()

        return df_symlots


# .. generate ohlcs for symlots
def get_ohlcs(
    MARKET: str, und_cts: list, savedf: bool, RUN_ON_PAPER: bool = False
) -> pd.DataFrame:

    ohlc_time = Timer(MARKET.lower() + "_ohlcs")
    ohlc_time.start()

    # ... set parameters from var.yml
    ibp = Vars(MARKET.upper())

    HOST, CID = ibp.HOST, ibp.CID
    if RUN_ON_PAPER:
        PORT = ibp.PAPER
    else:
        PORT = ibp.PORT

    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    if savedf:
        OP_FILENAME = "df_ohlcs.pkl"
    else:
        OP_FILENAME = ""

    with IB().connect(HOST, PORT, CID) as ib:
        df_ohlcs = ib.run(
            executeAsync(
                ib=ib,
                algo=ohlc,
                cts=und_cts,
                CONCURRENT=20,
                TIMEOUT=18,
                post_process=save_df,
                DATAPATH=DATAPATH,
                OP_FILENAME=OP_FILENAME,
                **{"DURATION": 365, "OHLC_DELAY": 15},
            )
        )
    ohlc_time.stop()

    return df_ohlcs


# .. generate underlyings df from symlots
def get_unds(
    MARKET: str, und_cts: list, savedf: bool = False, RUN_ON_PAPER: bool = False
) -> pd.DataFrame:

    und_time = Timer(MARKET.lower() + "_unds")
    und_time.start()

    # ... set parameters from var.yml
    ibp = Vars(MARKET.upper())

    HOST, CID = ibp.HOST, ibp.CID
    if RUN_ON_PAPER:
        PORT = ibp.PAPER
    else:
        PORT = ibp.PORT

    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    if savedf:
        OP_FILENAME = "df_unds.pkl"
    else:
        OP_FILENAME = ""

    with IB().connect(HOST, PORT, CID) as ib:
        df_unds = ib.run(
            executeAsync(
                ib=ib,
                algo=und,
                cts=und_cts,
                CONCURRENT=39,
                TIMEOUT=8,
                post_process=save_df,
                DATAPATH=DATAPATH,
                OP_FILENAME=OP_FILENAME,
                **{"FILL_DELAY": 8},
            )
        )
    und_time.stop()
    return df_unds


# .. get chains from symlots
def get_chains(
    MARKET: str, und_cts: list, savedf: bool, RUN_ON_PAPER: bool = False
) -> pd.DataFrame:

    chain_time = Timer(MARKET.lower() + "_chains")
    chain_time.start()

    # ... set parameters from var.yml
    ibp = Vars(MARKET.upper())

    HOST, CID = ibp.HOST, ibp.CID
    if RUN_ON_PAPER:
        PORT = ibp.PAPER
    else:
        PORT = ibp.PORT

    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    # ... for integrating lots to chains
    df_symlots = pd.read_pickle(DATAPATH.joinpath("df_symlots.pkl"))

    if savedf:
        OP_FILENAME = "df_chains.pkl"
    else:
        OP_FILENAME = ""

    with IB().connect(HOST, PORT, CID) as ib:
        df_chains = ib.run(
            executeAsync(
                ib=ib,
                algo=chain,
                cts=und_cts,
                CONCURRENT=44,
                TIMEOUT=5,
                post_process=save_df,
                DATAPATH=DATAPATH,
                OP_FILENAME=OP_FILENAME,
            )
        )

    # ..put lots into the chains
    df_chains = pd.read_pickle(DATAPATH.joinpath("df_chains.pkl"))

    if MARKET == "NSE":
        df_chains["expiryM"] = df_chains.expiry.apply(
            lambda d: d[:4] + "-" + d[4:6])
        cols1 = ["symbol", "expiryM"]
        df_chains = (
            df_chains.set_index(cols1)
            .join(df_symlots[cols1 + ["lot"]].set_index(cols1))
            .reset_index()
        )
        df_chains = df_chains.drop("expiryM", 1)
    else:
        df_chains["lot"] = 100

    # ...write back to pickle
    df_chains.to_pickle(DATAPATH.joinpath("df_chains.pkl"))

    chain_time.stop()


# .. get underlying margins
def get_und_margins(
    MARKET: str, und_cts: list, savedf: bool = False, RUN_ON_PAPER: bool = False
) -> pd.DataFrame:

    und_mgn_time = Timer(MARKET.lower() + "_und_margins")
    und_mgn_time.start()

    # ... set parameters from var.yml
    ibp = Vars(MARKET.upper())

    HOST, CID = ibp.HOST, ibp.CID
    if RUN_ON_PAPER:
        PORT = ibp.PAPER
    else:
        PORT = ibp.PORT

    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    df_symlots = pd.read_pickle(DATAPATH.joinpath("df_symlots.pkl"))

    if savedf:
        OP_FILENAME = "df_und_margins.pkl"
    else:
        OP_FILENAME = ""

    und_ords = (
        [MarketOrder("SELL", 100)] * len(df_symlots)
        if MARKET.upper() == "SNP"
        else [MarketOrder("SELL", q) for q in df_symlots.lot]
    )

    und_cos = [(c, o) for c, o in zip(und_cts, und_ords)]

    with IB().connect(HOST, PORT, CID) as ib:
        df_und_margins = ib.run(
            executeAsync(
                ib=ib,
                algo=margin,
                cts=und_cos,
                CONCURRENT=100,
                TIMEOUT=8,
                post_process=save_df,
                DATAPATH=DATAPATH,
                OP_FILENAME=OP_FILENAME,
            )
        )

    und_mgn_time.stop()
    return df_und_margins


# . generate and qualify options
def qualify_opts(
    MARKET: str,
    RUN_ON_PAPER: bool = True,  # Use PAPER account
    REUSE: bool = True,  # Reuse old df_opts, if available
    OP_FILENAME: str = "qopts.pkl",  # Filename to save the qualified options
):
    ibp = Vars(MARKET.upper())  # IB Parameters from var.yml

    # ... start the timer
    opts_time = Timer("qualify all options")
    opts_time.start()

    HOST, CID = ibp.HOST, ibp.CID

    if RUN_ON_PAPER:
        print(f"\nQualifying {MARKET} raw options using Paper account\n")
        PORT = ibp.PAPER
    else:
        PORT = ibp.PORT

    LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")
    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    # * SETUP LOGS AND CLEAR THEM
    LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_qopts.log")
    util.logToFile(path=LOGFILE, level=30)
    with open(LOGFILE, "w"):
        pass

    # * LOAD FILES
    df_chains = pd.read_pickle(DATAPATH.joinpath("df_chains.pkl"))

    if REUSE:  # remove existing option symbols from chains
        try:
            qopts = pd.read_pickle(DATAPATH.joinpath(OP_FILENAME))
            available_opts = {q.symbol for q in qopts}
            df_chains = df_chains[~df_chains.symbol.isin(available_opts)]
        except FileNotFoundError:
            print(
                f"\n{OP_FILENAME} for {MARKET} is not found!\n"
                + f"All chains will be qualified!!\n"
            )
    else:
        qopts = pd.Series([], dtype=object, name="contract")

    if df_chains.empty:
        print(
            f"\nqopts.pkl has all df_chains symbols and so it is not recreated!"
            + f"\nRun qualify_opts with REUSE = False to recreate entirely new qopts.pkl\n"
        )
        return qopts

    df_ch = pd.concat(
        (df_chains.assign(right="P"), df_chains.assign(right="C")), ignore_index=True
    )
    # ... build the options
    opts = [
        Option(s, e, k, r, x)
        for s, e, k, r, x in zip(
            df_ch.symbol,
            df_ch.expiry,
            df_ch.strike,
            df_ch.right,
            ["NSE" if MARKET.upper() == "NSE" else "SMART"] * len(df_ch),
        )
    ]

    # ... generate new options
    with IB().connect(HOST, PORT, CID) as ib:
        new_qopts = ib.run(
            executeAsync(
                ib=ib,
                algo=qualify,
                cts=opts,
                CONCURRENT=100,
                TIMEOUT=5,
                post_process=save_df,
                DATAPATH=DATAPATH,
                OP_FILENAME="z_new_qopts_temp.pkl",
            )
        )

    # ... merge the qualified options
    qopts = pd.concat([qopts, new_qopts], ignore_index=True).drop_duplicates()

    # ... pickle and clean
    qopts.to_pickle(DATAPATH.joinpath("qopts.pkl"))
    try:
        os.remove(DATAPATH.joinpath("z_new_qopts_temp.pkl"))
    except OSError as e:
        print(f"No temp file to remove!! Error: {e}")

    opts_time.stop()

    return qopts


if __name__ == "__main__":

    # * USER INTERFACE

    # . first... get first user input for market
    mkt_dict = {1: "NSE", 2: "SNP"}
    mkt_ask_range = [i + 1 for i in list(range(len(mkt_dict)))]
    mkt_ask = "Create base data for:\n"
    mkt_ask = mkt_ask + "1) NSE\n"
    mkt_ask = mkt_ask + "2) SNP\n"

    while True:
        data = input(mkt_ask)  # check for int in input
        try:
            mkt_int = int(data)
        except ValueError:
            print("\nI didn't understand what you entered. Try again!\n")
            continue  # Loop again
        if not mkt_int in mkt_ask_range:
            print(f"\nWrong number! choose between {mkt_ask_range}...")
        else:
            MARKET = mkt_dict[mkt_int]
            break  # success and exit loop

    # . second ... ask if the routines need to run on paper
    paper_ip = input("\nShould Engine run on `Paper`? (Y/N) ").lower()
    if paper_ip == "y":
        RUN_ON_PAPER = True
    else:
        RUN_ON_PAPER = False

    # . third ... check if the existing pickles need to be deleted
    delete_ip = input("\n\nDelete Engine pickles? (Y/N) ").lower()
    if delete_ip == "y":
        DELETE_PICKLES = True
    else:
        DELETE_PICKLES = False

    # . fourth ... check if the qopts.pkl are to be deleted
    reuse_qots_ip = input(
        "\n\nRe-use `qopts.pkl`? (Y/N) \n...Note: regenerting qopts takes time!\n"
    ).lower()
    if reuse_qots_ip == "y":
        REUSE = True
    else:
        REUSE = False

    # * SET THE VARIABLES

    ibp = Vars(MARKET.upper())  # IB Parameters from var.yml

    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    # * SETUP LOGS AND CLEAR THEM
    LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")
    LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_base.log")
    util.logToFile(path=LOGFILE, level=30)
    with open(LOGFILE, "w"):
        pass

    # * DELETE UNNECESSARY FILES
    if DELETE_PICKLES:
        for f in [
            "df_symlots.pkl",
            "df_unds.pkl",
            "df_ohlcs.pkl",
            "df_chains.pkl",
            "df_und_margins.pkl",
            "z_new_qopts_temp.pkl",
        ]:
            try:
                os.remove(DATAPATH.joinpath(f))
            except OSError as e:
                print(f"\nCannot remove file. {e}")

    # .. start the timer
    all_time = Timer("Engine")
    all_time.start()

    df_symlots = get_symlots(MARKET=MARKET, RUN_ON_PAPER=RUN_ON_PAPER)

    und_cts = list(df_symlots.contract.unique())

    df_unds = get_unds(
        MARKET=MARKET, RUN_ON_PAPER=RUN_ON_PAPER, und_cts=und_cts, savedf=True
    )

    df_ohlcs = get_ohlcs(
        MARKET=MARKET, RUN_ON_PAPER=RUN_ON_PAPER, und_cts=und_cts, savedf=True
    )

    df_chains = get_chains(
        MARKET=MARKET, RUN_ON_PAPER=RUN_ON_PAPER, und_cts=und_cts, savedf=True
    )

    df_und_margins = get_und_margins(
        MARKET=MARKET, RUN_ON_PAPER=RUN_ON_PAPER, und_cts=und_cts, savedf=True
    )

    qopts = qualify_opts(
        MARKET=MARKET, RUN_ON_PAPER=RUN_ON_PAPER, REUSE=REUSE, OP_FILENAME="qopts.pkl"
    )

    all_time.stop()
