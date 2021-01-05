# ** SETUP

# Prevent spurious problems for try...
# pyright: reportUnboundVariable=false

# .Imports

import asyncio
import math
import os
import pathlib
from collections import defaultdict, namedtuple
from datetime import datetime
from io import StringIO
from typing import Callable, Coroutine, Union

import numpy as np
import pandas as pd
import requests
from ib_insync import IB, Contract, MarketOrder, Option, util
from tqdm import tqdm

from opts import make_opts
from support import (Timer, Vars, calcsdmult_df, get_dte, get_prob, quick_pf,
                     yes_or_no)

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
BAR_FORMAT = "{desc:<10}{percentage:3.0f}%|{bar:25}{r_bar}{bar:-10b}"

# ** SINGLE FUNCTIONS
# * Independent functions - ready to be called.

# .qualify
async def qualify(ib: IB, c: Union[pd.Series, list, tuple,
                                   Contract]) -> pd.Series:

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
async def ohlc(ib: IB,
               c,
               WHAT_TO_SHOW: str = 'TRADES',
               DURATION: int = 365,
               BAR_SIZE = "1 day",
               OHLC_DELAY: int = 5) -> pd.DataFrame:

    "Gets OHLC history for stocks / options in UTC date"

    if isinstance(c, tuple):
        c = c[0]

    DUR = str(DURATION) + " D"

    ohlc = await ib.reqHistoricalDataAsync(
        contract=c,
        endDateTime=datetime.now(),
        durationStr=DUR,
        barSizeSetting=BAR_SIZE,
        whatToShow=WHAT_TO_SHOW,
        useRTH=True,
        formatDate=2,  # UTC format
    )
    await asyncio.sleep(OHLC_DELAY)
    df = util.df(ohlc)

    # re-cast OHLC for options
    if WHAT_TO_SHOW == 'BID_ASK':

        # get the grouped values for the option
        try:
            g = df.groupby("date")
        except AttributeError:
            df = pd.DataFrame({'date': pd.NaT, 'conId': c.conId, 
                               'symbol': c.symbol, 'localSymbol': c.localSymbol, 
                               'strike': c.strike, 'right': c.right, 
                               'expiry': c.lastTradeDateOrContractMonth, 
                               'bid': np.nan, 'ask': np.nan}, index=range(1))
            
            return df # !!! ABORTED OHLC

        op = g.open.first()
        hi = g.high.max()
        lo = g.low.min()
        cl = g.close.last()
        vol = g.volume.sum()
        avg = g.average.mean()
        bc = g.barCount.sum()

        df1 = pd.DataFrame({"date": df.date.unique()})

        df = df1.assign(
            open=df1["date"].map(op),
            high=df1["date"].map(hi),
            low=df1["date"].map(lo),
            close=df1["date"].map(cl),
            volume=df1["date"].map(vol),
            average=df1["date"].map(avg),
            barCount=df1["date"].map(bc),
        )
        
        df = df.drop(['open', 'close', 'average', 'barCount', 'volume'], 1)
        df.rename(columns={'high': 'ask', 'low': 'bid'}, 
                            inplace=True)

        df.insert(0, "localSymbol", c.localSymbol)
        df.insert(1, "strike", c.strike)
        df.insert(2, "right", c.right)
        df.insert(3, "expiry", c.lastTradeDateOrContractMonth)

    df.insert(0, "conId", c.conId)
    df.insert(1, "symbol", c.symbol)

    return df


# .Underlying
async def und(ib: IB, c, FILL_DELAY=8) -> pd.DataFrame:
    """Use CONCURRENT=40, TIMEOUT=8 for optimal executeAsync results"""

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

    hv = None
    try:
        hv = df3.histVolatility.iloc[0]
        df3.impliedVolatility  # check if iv is available
    except AttributeError:
        print(f"\niv missing for {df3.symbol.to_list()}")
        if hv:
            df3["impliedVolatility"] = hv
            print("...iv is replaced by historical volatility")
        else:
            df3["impliedVolatility"] = np.nan

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
    df2 = (pd.merge(
        pd.DataFrame(df1.expirations[0], columns=["expiry"]).assign(key=1),
        pd.DataFrame(df1.strikes[0], columns=["strike"]).assign(key=1),
        on="key",
    ).merge(df1.assign(key=1)).rename(columns={
        "tradingClass": "symbol",
        "multiplier": "mult"
    })[["symbol", "expiry", "strike", "exchange", "mult"]])

    # Replace tradingclass to reflect correct symbol name of 9 characters
    df2 = df2.assign(symbol=df2.symbol.str[:9])

    # convert & to %26
    df2 = df2.assign(symbol=df2.symbol.str.replace("&", "%26"))

    # convert symbols - friendly to IBKR
    df2 = df2.assign(symbol=df2.symbol.str.slice(0, 9))
    ntoi = {
        "M%26M": "MM",
        "M%26MFIN": "MMFIN",
        "L%26TFH": "LTFH",
        "NIFTY": "NIFTY50"
    }
    df2.symbol = df2.symbol.replace(ntoi)

    # Get the dte
    df2["dte"] = df2.expiry.apply(get_dte)
    df2 = df2[df2.dte > 0]  # Remove negative dtes
    # Make 0 dte positive to avoid sqrt errors
    df2.loc[df2.dte == 0, "dte"] = 1

    return df2


# .Price
async def price(ib: IB, co, **kwargs) -> pd.DataFrame:
    """Gives price and iv (if available)
    Args:
        ib: IB connection
        co: a qualified contract
    Returns:
        DataFrame
    Usage:
        df = ib.run(price(ib, co, **{"FILL_DELAY"}: 8))

    Note: For optimal concurrency:
            CONCURRENT=40 and TIMEOUT=8
            gives ~250 contract prices/min"""

    df_empty = pd.DataFrame({
        "symbol": {},
        "secType": {},
        "localSymbol": {},
        "conId": {},
        "strike": {},
        "expiry": {},
        "right": {},
        "contract": {},
        "time": {},
        "greek": {},
        "bid": {},
        "ask": {},
        "close": {},
        "last": {},
        "price": {},
        "iv": {},
    })

    try:
        FILL_DELAY = kwargs["FILL_DELAY"]
    except KeyError as ke:
        print(
            f"\nWarning: No FILL_DELAY supplied! 5.5 second default is taken\n"
        )
        FILL_DELAY = 5.5

    try:

        if isinstance(co, tuple):
            c = co[0]
        else:
            c = co

        df = (util.df([c]).iloc[:, :6].rename(
            columns={"lastTradeDateOrContractMonth": "expiry"}))

    except (TypeError, AttributeError, ValueError) as err:
        print(f"\nError: contract {co} supplied is incorrect!" + f"\n{err}" +
              f"\n... and empty df will be returned !!")

        df = df_empty

        return df  # ! Aborted return with empty df for contract error

    tick = ib.reqMktData(c, genericTickList="106")

    await asyncio.sleep(FILL_DELAY)

    df = df.assign(localSymbol=c.localSymbol, contract=c)

    try:
        dfpr = util.df([tick])

        if dfpr.modelGreeks[0] is None:
            iv = dfpr.impliedVolatility
        else:
            iv = dfpr.modelGreeks[0].impliedVol

        df = df.assign(
            time=dfpr.time,
            greeks=dfpr.modelGreeks,
            bid=dfpr.bid,
            ask=dfpr.ask,
            close=dfpr["close"],
            last=dfpr["last"],
            price=dfpr["last"].combine_first(dfpr["close"]),
            iv=iv,
        )

    except AttributeError as e:

        print(
            f"\nError in {c.localSymbol}: {e}. df will have no price and iv!\n"
        )

        df = df.assign(
            time=np.nan,
            greeks=np.nan,
            bid=np.nan,
            ask=np.nan,
            close=np.nan,
            last=np.nan,
            price=np.nan,
            iv=np.nan,
        )

    ib.cancelMktData(c)

    return df


async def qpCoro(ib: IB, contract: Contract, **kwargs) -> pd.DataFrame:
    """Coroutine for quick price from market | history"""

    try:
        FILL_DELAY = kwargs["FILL_DELAY"]
    except KeyError as ke:
        print(
            f"\nWarning: No FILL_DELAY supplied! 5.5 second default is taken\n"
        )
        FILL_DELAY = 5.5
    
    if isinstance(contract, tuple):
        contract = contract[0]
    
    df_mktpr = await price(ib, contract, **kwargs)
    
    async def histCoro():
        
        result = defaultdict(dict)
        
        try:
            ticks = await asyncio.wait_for(ib.reqHistoricalTicksAsync(
                            contract=contract,
                            startDateTime="",
                            endDateTime=datetime.now(),
                            numberOfTicks=1,
                            whatToShow="Bid_Ask",
                            useRth=False,
                            ignoreSize=False), timeout=None)

        except asyncio.TimeoutError:
            tick = namedtuple('tick', ['time', 'priceBid', 'priceAsk'])
            ticks = [tick(time=pd.NaT, priceBid=np.nan, priceAsk=np.nan)]

        # extract bid and ask price, if available!
        try:
            bid_ask = ticks[-1]  # bid ask is not availble for Index securities!
            result["bid"] = bid_ask.priceBid
            result["ask"] = bid_ask.priceAsk
            result["batime"] = bid_ask.time

        except IndexError:
            result["bid"] = np.nan
            result["ask"] = np.nan
            result["batime"] = pd.NaT
        
        return result
    
    # bid/ask with -1.0 as market is not open
    if (df_mktpr.bid.iloc[0] == -1.0) or (df_mktpr.ask.iloc[0] == -1.0):
        result = await histCoro()
        
        df_pr = df_mktpr.assign(batime = result['batime'],
                    bid = result['bid'],
                    ask = result['ask'])
    else:
        df_mktpr['batime'] = df_mktpr['time']
        df_pr = df_mktpr
        
    # use bid-ask avg if last price is not available
    df_pr = df_pr.assign(price=df_pr["last"]\
                 .combine_first(df_pr[["bid", "ask"]]\
                 .mean(axis=1)))
    
    df_pr = df_pr.sort_values(['right', 'strike'], ascending=[True, False])
    
    return df_pr

async def qpAsync(ib:IB, contracts, **kwargs) -> pd.DataFrame:
    """Quick Price with bid-ask for a number of contracts"""
    
    if hasattr(contracts, '__iter__'):
        tasks = [qpCoro(ib=ib, contract=contract, **kwargs) for contract in contracts]
    else:
        tasks = [qpCoro(ib=ib, contract=contracts, **kwargs)]
        
    df_prs = [await res for res in tqdm(asyncio.as_completed(tasks), 
                      desc=f"opt price:",
                      bar_format=BAR_FORMAT,
                      ncols=80,
                      total=len(tasks))]

    df = pd.concat(df_prs, ignore_index=True)

    return df

# .Margin coroutine
async def margin(ib: IB, co, **kwargs) -> pd.DataFrame:
    """Optimal execAsync: CONCURRENT=200 with TIMEOUT=FILL_DELAY=5.5"""

    try:
        FILL_DELAY = kwargs["FILL_DELAY"]
    except KeyError as ke:
        print(
            f"\nWarning: No FILL_DELAY supplied!. 1.5 second default is taken\n"
        )
        FILL_DELAY = 1.5

    TEMPL_PATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "template",
                                             "df_margin.pkl")
    df_empty = pd.read_pickle(TEMPL_PATH)

    try:
        ct, o = pre_process(co)
    except ValueError as ve:
        print(
            f"\nError: {co} co supplied is incorrect! It should be a tuple(ct, o)\n"
        )
        df = df_empty

    async def wifAsync(ct, o):
        wif = ib.whatIfOrderAsync(ct, o)
        await asyncio.sleep(FILL_DELAY)
        return wif

    wif = await wifAsync(ct, o)

    if wif.done():

        res = wif.result()

        try:

            df = (util.df([ct]).iloc[:, :6].rename(
                columns={"lastTradeDateOrContractMonth": "expiry"}))

            df = df.join(
                util.df(
                    [res])[["initMarginChange", "maxCommission",
                            "commission"]])

        except TypeError as e:

            print(f"\nError: Unknown type of contract: {ct}, order: {o}" +
                  f" \n...in margin wif: {wif} !\n" +
                  f"   \n..... giving error{e}")

            df = df_empty

        except IndexError as e:

            print(f"\nError: Index error for contract: {ct}, order: {o}" +
                  f" \n...in margin wif: {wif} \n" +
                  f"   \n..... giving error{e}")

            df = df_empty

    else:

        print(
            f"\nError: wif could not complete for contract: {ct.localSymbol}" +
            f"\nTry by increasing FILL_DELAY from > {FILL_DELAY} secs\n")

        df = df_empty

    # post-processing df
    df = df.assign(secType=ct.secType,
                   conId=ct.conId,
                   localSymbol=ct.localSymbol,
                   symbol=ct.symbol)
    df = df.assign(
        comm=df[["commission", "maxCommission"]].min(axis=1),
        margin=df.initMarginChange.astype("float"),
    )

    # Correct unrealistic margin and commission
    df = df.assign(
        margin=np.where(df.margin > 1e7, np.nan, df.margin),
        comm=np.where(df.comm > 1e7, np.nan, df.comm),
    )

    df = df[[
        "conId",
        "secType",
        "symbol",
        "strike",
        "right",
        "expiry",
        "localSymbol",
        "margin",
        "comm",
    ]]

    return df


# ** EXECUTION
# * Core engine that processes functions and delivers results

# .preprocessing data for the core engine


def pre_process(cts):
    """Generates tuples for input to the engine"""

    try:
        symbol = cts.symbol
        output = ((cts, None), )

    except AttributeError as ae1:  # it's an iterable!
        try:
            symbols = [c.symbol for c in cts]

            if len(symbols) == 1:
                output = ((cts[0], None), )
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
            c.symbol + c.lastTradeDateOrContractMonth[-4:] + c.right +
            str(c.strike) + ".." for c in cts
        ]

    except TypeError as te:  # single non-iterable element
        if cts != "":  # not empty!
            output = (cts.symbol + cts.lastTradeDateOrContractMonth[-4:] +
                      cts.right + str(cts.strike))
        else:
            output = cts

    except AttributeError as ae1:  # multiple (p, s) combination
        try:
            output = [
                c[0].symbol + c[0].lastTradeDateOrContractMonth[-4:] +
                c[0].right + str(c[0].strike) + ".." for c in cts
            ]
        except TypeError as te2:
            output = (cts[0].symbol +
                      cts[0].lastTradeDateOrContractMonth[-4:] + cts[0].right +
                      str(cts[0].strike))

    return output


# .the core engine
async def executeAsync(
    ib: IB(),
    algo: Callable[..., Coroutine],  # coro name
    cts: Union[Contract, pd.Series, list, tuple],  # list of contracts
    CONCURRENT: int = 44,  # to prevent overflows put 44 * (TIMEOUT-1)
    TIMEOUT: None = None,  # if None, no progress messages shown
    post_process: Callable[[set, pathlib.Path, str],
                           pd.DataFrame] = None,  # If checkpoint is needed
    DATAPATH: pathlib.Path = None,  # Necessary for post_process
    OP_FILENAME: str = "",  # output file name
    SHOW_TQDM: bool = True,  # Show tqdm bar instead of individual messages
    REUSE: bool = False,  # Reuse the OP_FILENAME supplied
    **kwargs,
) -> pd.DataFrame:

    tasks = set()
    results = set()
    remaining = pre_process(cts)
    last_len_tasks = 0  # tracking last length for error catch

    # Set pbar
    if SHOW_TQDM:
        pbar = tqdm(
            total=len(remaining),
            desc=f"{algo.__name__}: ",
            bar_format=BAR_FORMAT,
            ncols=80,
            leave=False,
        )

    # Get the results
    while len(remaining):

        # Tasks limited by concurrency
        if len(remaining) <= CONCURRENT:

            tasks.update(
                asyncio.create_task(algo(ib, c, **kwargs), name=make_name(c))
                for c in remaining)

        else:

            tasks.update(
                asyncio.create_task(algo(ib, c, **kwargs), name=make_name(c))
                for c in remaining[:CONCURRENT])

        # Execute tasks
        while len(tasks):

            done, tasks = await asyncio.wait(tasks,
                                             timeout=TIMEOUT,
                                             return_when=asyncio.ALL_COMPLETED)

            # Remove dones from remaining
            done_names = [d.get_name() for d in done]
            remaining = [
                c for c in remaining if make_name(c) not in done_names
            ]

            # Update results and checkpoint
            results.update(done)

            # Checkpoint the results
            if post_process:

                output = post_process(
                    results=results,
                    DATAPATH=DATAPATH,
                    REUSE=REUSE,
                    LAST_RUN=False,
                    OP_FILENAME=OP_FILENAME,
                )

                if not output.empty:
                    REUSE = False  # for second run onwards

            else:
                output = results

            if TIMEOUT:

                if remaining:

                    if SHOW_TQDM:
                        pbar.update(len(done))

                    else:
                        print(
                            f"\nDone {algo.__name__} for {done_names[:2]} {len(results)} out of {len(cts)}. Pending {[make_name(c) for c in remaining][:2]}"
                        )

                # something wrong. Task is not progressing
                if (len(tasks) == last_len_tasks) & (len(tasks) > 0):
                    print(
                        f"\n @ ALERT @: Tasks failing. Pending {len(tasks)} tasks such as {[t.get_name() for t in tasks][:3]}... will be killed in 5 seconds !\n"
                    )
                    dn, pend = await asyncio.wait(tasks, timeout=5.0)
                    if len(dn) > 0:
                        results.update(dn)

                    tasks.difference_update(dn)
                    tasks.difference_update(pend)

                    pend_names = [p.get_name() for p in pend]
                    # remove pending from remaining
                    remaining = [
                        c for c in remaining if make_name(c) not in pend_names
                    ]

                # re-initialize last length of tasks
                last_len_tasks = len(tasks)

    # Make the final output, based on REUSE status

    if OP_FILENAME:
        df = post_process(
            results=set(),  # Empty dataset
            DATAPATH=DATAPATH,
            REUSE=REUSE,
            LAST_RUN=True,
            OP_FILENAME=OP_FILENAME,
        )
    else:
        df = output

    if SHOW_TQDM:

        pbar.update(len(done))
        pbar.refresh()
        pbar.close()

    return df


# .Process output into dataframes
def post_df(
    results: set,
    DATAPATH: pathlib.Path,
    REUSE: bool,
    LAST_RUN: bool,
    OP_FILENAME: str = "",
) -> pd.DataFrame():

    if results:
        df = pd.concat([r.result() for r in results if r], ignore_index=True)

        if OP_FILENAME:

            if REUSE:

                # load the existing file

                try:
                    df_old = pd.read_pickle(DATAPATH.joinpath(OP_FILENAME))

                    # Save old temporarily
                    df_old.to_pickle(DATAPATH.joinpath("z_temp_" +
                                                       OP_FILENAME))

                except FileNotFoundError:
                    pass

            df.to_pickle(DATAPATH.joinpath(OP_FILENAME))

    else:

        if LAST_RUN:  # Merge new and old df (if available)

            if OP_FILENAME:

                try:
                    df_old = pd.read_pickle(
                        DATAPATH.joinpath("z_temp_" + OP_FILENAME))
                    # cleanup temp file
                    os.remove(DATAPATH.joinpath("z_temp_" + OP_FILENAME))

                except FileNotFoundError:
                    df_old = pd.DataFrame([])

                df_new = pd.read_pickle(DATAPATH.joinpath(OP_FILENAME))
                df = df_new.append(df_old).reset_index(drop=True)
                df.to_pickle(DATAPATH.joinpath(OP_FILENAME))

        else:
            df = pd.DataFrame([])  # results are not yet ready!

    return df


# ** ACTIVITIES
# * Function calls to achieve specific results


def get_nse() -> pd.DataFrame:
    """Make nse symbols, qualify them and put into a dataframe"""

    url = "https://www1.nseindia.com/content/fo/fo_mktlots.csv"

    try:
        req = requests.get(url)
        if req.status_code == 404:
            print(f"\n{url} URL contents not correct. 404 error!")
        df_symlots = pd.read_csv(StringIO(req.text))
    except requests.ConnectionError as e:
        print(f"Connection Error {e}")
    except pd.errors.ParserError as e:
        print(f"Parser Error {e}")

    df_symlots = df_symlots[list(df_symlots)[1:5]] 

    # strip whitespace from columns and make it lower case
    df_symlots.columns = df_symlots.columns.str.strip().str.lower()

    # strip all string contents of whitespaces
    df_symlots = df_symlots.applymap(lambda x: x.strip()
                                     if type(x) is str else x)

    # remove 'Symbol' row
    df_symlots = df_symlots[df_symlots.symbol != "Symbol"]

    # melt the expiries into rows
    df_symlots = df_symlots.melt(id_vars=["symbol"],
                                 var_name="expiryM",
                                 value_name="lot").dropna()

    # remove rows without lots
    df_symlots = df_symlots[~(df_symlots.lot == "")]

    # convert expiry to period
    df_symlots = df_symlots.assign(expiryM=pd.to_datetime(
        df_symlots.expiryM, format="%b-%y").dt.to_period("M").astype("str"))

    # convert lots to integers
    df_symlots = df_symlots.assign(
        lot=pd.to_numeric(df_symlots.lot, errors="coerce"))

    # convert & to %26
    df_symlots = df_symlots.assign(
        symbol=df_symlots.symbol.str.replace("&", "%26"))

    # convert symbols - friendly to IBKR
    df_symlots = df_symlots.assign(symbol=df_symlots.symbol.str.slice(0, 9))
    ntoi = {
        "M%26M": "MM",
        "M%26MFIN": "MMFIN",
        "L%26TFH": "LTFH",
        "NIFTY": "NIFTY50"
    }
    df_symlots.symbol = df_symlots.symbol.replace(ntoi)

    # differentiate between index and stock
    df_symlots.insert(
        1, "secType",
        np.where(df_symlots.symbol.str.contains("NIFTY"), "IND", "STK"))

    df_symlots["exchange"] = "NSE"
    df_symlots["currency"] = "INR"
    df_symlots["contract"] = [
        Contract(symbol=symbol,
                 secType=secType,
                 exchange=exchange,
                 currency=currency)
        for symbol, secType, exchange, currency in zip(
            df_symlots.symbol,
            df_symlots.secType,
            df_symlots.exchange,
            df_symlots.currency,
        )
    ]

    return df_symlots


# .SNP df
def get_snp(RUN_ON_PAPER: bool = True) -> pd.DataFrame():
    """Generate symlots for SNP 500 weeklies + those in portfolio as a DataFrame"""

    MARKET = "SNP"

    # ... set parameters from var.yml

    ibp = Vars(MARKET.upper())
    HOST, CID = ibp.HOST, ibp.CID
    if RUN_ON_PAPER:
        PORT = ibp.PAPER
    else:
        PORT = ibp.PORT

    # Get the weeklies
    dls = "http://www.cboe.com/products/weeklys-options/available-weeklys"

    try:
        data = pd.read_html(dls)
    except Exception as e:
        print(f"Error: {e}")
    """ 
    # * NOT WORKING FROM 12-DEC-2020. CBOE XLS FORMAT CHANGED!!
    df_ix = pd.concat([data[i].loc[:, :0] for i in range(1, 3)],
                      ignore_index=True).rename(columns={0: "symbol"})
    df_ix = df_ix[df_ix.symbol.apply(len) <= 5]
    df_ix["secType"] = "IND"
    df_ix["exchange"] = "CBOE"

    df_eq = data[4].iloc[:, :1].rename(columns={0: "symbol"})
    df_eq = df_eq[df_eq.symbol.apply(len) <= 5]
    df_eq["secType"] = "STK"
    df_eq["exchange"] = "SMART"

    df_weeklies = pd.concat([df_ix, df_eq], ignore_index=True) """

    df = data[0].rename(columns={"STOCKSYMBOL": "symbol"})[["symbol"]]
    df_weeklies = pd.concat(
        [
            df.assign(secType="STK", exchange="SMART"),
            df.assign(secType="IND", exchange="CBOE"),
        ],
        ignore_index=True,
    )

    df_weeklies = df_weeklies.assign(
        symbol=df_weeklies.symbol.str.replace("[^a-zA-Z]", ""))

    # Generate the snp 500s
    try:
        s500 = list(
            pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                header=0,
                match="Symbol",
            )[0].loc[:, "Symbol"])
    except Exception as e:
        print(f"Error: {e}")

    # without dot in symbol
    snp500 = [s.replace(".", "") if "." in s else s for s in s500]

    # Keep only equity weeklies that are in S&P 500, and all indexes in the weeklies
    df_symlots = df_weeklies[((df_weeklies.secType == "STK") &
                              (df_weeklies.symbol.isin(snp500)))
                             | (df_weeklies.secType == "IND")].reset_index(
                                 drop=True)

    with IB().connect(HOST, PORT, CID) as ib:
        pf = quick_pf(ib)

        # to prevent first TimeoutError()
        ib.disconnect()
        IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

    # Additional symbols
    more_syms = set(pf.symbol) - set(df_symlots.symbol)
    more_syms = more_syms | set(ibp.SPECIALS)

    df_syms = pd.concat(
        [
            pd.DataFrame({
                "symbol": list(more_syms),
                "secType": "STK",
                "exchange": "SMART"
            }),
            pd.DataFrame({
                "symbol": list(more_syms),
                "secType": "IND",
                "exchange": "SMART"
            }),
            df_symlots,
        ],
        ignore_index=False,
    )

    # Append other symlot fields
    df_syms = (df_syms.assign(
        expiry=None, lot=100,
        currency="USD").drop_duplicates().reset_index(drop=True))

    return df_syms


# .. generate symbols and lots
def get_symlots(MARKET: str, RUN_ON_PAPER: bool = False) -> pd.DataFrame:

    u_qual_time = Timer(MARKET.lower() + " symlot qualification")
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
    df_symlots = (get_nse() if MARKET.upper() == "NSE" else get_snp(
        RUN_ON_PAPER=RUN_ON_PAPER))

    # ... remove YAML blacklisted contracts
    df_symlots = df_symlots[~df_symlots.symbol.isin(ibp.BLACKLIST)]

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

    with IB().connect(HOST, PORT, CID) as ib:
        und_cts = ib.run(
            executeAsync(
                ib=ib,
                algo=qualify,
                cts=raw_unds,
                CONCURRENT=100,
                TIMEOUT=5,
                post_process=post_df,
            ))

        # to prevent first TimeoutError()
        ib.disconnect()
        IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

    df_symlots = df_symlots.assign(
        contract=df_symlots.symbol.map({u.symbol: u
                                        for u in und_cts}))

    # remove any NaN
    df_symlots = df_symlots.dropna(subset=["contract"])

    # remove beyond SNP duplicates with secType
    df_symlots = df_symlots[df_symlots.secType == df_symlots.contract.apply(
        lambda x: x.secType)].reset_index(drop=True)

    df_symlots.to_pickle(DATAPATH.joinpath("df_symlots.pkl"))

    u_qual_time.stop()

    return df_symlots


# .. generate ohlcs for symlots
def get_ohlcs(MARKET: str,
              und_cts: list,
              SAVE: bool,
              RUN_ON_PAPER: bool = False) -> pd.DataFrame:

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

    if SAVE:
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
                post_process=post_df,
                DATAPATH=DATAPATH,
                OP_FILENAME=OP_FILENAME,
                **{
                    "DURATION": 365,
                    "OHLC_DELAY": 15
                },
            ))
    ohlc_time.stop()

    return df_ohlcs


# .. generate underlyings df from symlots
def get_unds(MARKET: str,
             und_cts: list,
             SAVE: bool = False,
             RUN_ON_PAPER: bool = False) -> pd.DataFrame:

    und_time = Timer(MARKET.lower() + " underlyings")
    und_time.start()

    # ... set parameters from var.yml
    ibp = Vars(MARKET.upper())

    HOST, CID = ibp.HOST, ibp.CID
    if RUN_ON_PAPER:
        PORT = ibp.PAPER
    else:
        PORT = ibp.PORT

    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())
    df_symlots = pd.read_pickle(DATAPATH.joinpath("df_symlots.pkl"))

    # Reduce df_symlots to und_cts contracts
    df_symlots = df_symlots[df_symlots.symbol.isin([c.symbol for c in und_cts
                                                    ])].reset_index(drop=True)

    # remove duplicate symbols in df_symlots (duplicates due to NSE lot changes)
    df_symlots = df_symlots.drop_duplicates(["symbol"]).reset_index(drop=True)

    if SAVE:
        OP_FILENAME = "df_unds.pkl"
    else:
        OP_FILENAME = ""

    # Quicken unds for single symbol
    if len(und_cts) > 1:
        DLY = 8
    else:
        DLY = 5

    with IB().connect(HOST, PORT, CID) as ib:

        ib.client.setConnectOptions('PACEAPI')

        df_unds = ib.run(
            executeAsync(
                ib=ib,
                algo=und,
                cts=und_cts,
                CONCURRENT=40,
                TIMEOUT=DLY,
                post_process=post_df,
                DATAPATH=DATAPATH,
                OP_FILENAME="",
                **{"FILL_DELAY": DLY},
            ))
        ib.disconnect()
        IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

    und_ords = ([MarketOrder("SELL", 100)] * len(df_symlots) if MARKET.upper()
                == "SNP" else [MarketOrder("SELL", q) for q in df_symlots.lot])

    und_cos = [(c, o) for c, o in zip(und_cts, und_ords)]
    """ df_und_margins = get_margins(
        MARKET=MARKET, cos=und_cos, msg='und margins') """

    with IB().connect(HOST, PORT, CID) as ib:
        df_und_margins = ib.run(
            executeAsync(
                ib=ib,
                algo=margin,
                cts=und_cos,
                CONCURRENT=200,
                TIMEOUT=15,
                post_process=post_df,
                DATAPATH=DATAPATH,
                OP_FILENAME="",
                **{"FILL_DELAY": 14.5},
            ))

    df_und_margins[["conId", "margin", "comm"]]
    df_unds = df_unds.assign(conId=[c.conId for c in df_unds.contract])
    df_unds = (df_unds.set_index("conId").join(
        df_und_margins[["conId", "margin",
                        "comm"]].set_index("conId")).reset_index())

    df_unds = df_unds[df_unds.undPrice.notna()] # Remove unds without undPrice

    if OP_FILENAME:
        df_unds.to_pickle(DATAPATH.joinpath("df_unds.pkl"))

    und_time.stop()
    return df_unds


# .. get chains from symlots
def get_chains(MARKET: str,
               und_cts: list,
               SAVE: bool,
               RUN_ON_PAPER: bool = False) -> pd.DataFrame:

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
    df_symlots = df_symlots[df_symlots.symbol.isin([c.symbol for c in und_cts
                                                    ])].reset_index(drop=True)

    if SAVE:
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
                post_process=post_df,
                DATAPATH=DATAPATH,
                OP_FILENAME=OP_FILENAME,
            ))

    # ..put lots into the chains
    # df_chains = pd.read_pickle(DATAPATH.joinpath("df_chains.pkl"))

    if MARKET == "NSE":
        df_chains["expiryM"] = df_chains.expiry.apply(
            lambda d: d[:4] + "-" + d[4:6])
        cols1 = ["symbol", "expiryM"]
        df_chains = (df_chains.set_index(cols1).join(
            df_symlots[cols1 + ["lot"]].set_index(cols1)).reset_index())
        df_chains = df_chains.drop("expiryM", 1)
    else:
        df_chains["lot"] = 100

    # ..remove NaNs
    df_chains = df_chains.dropna().reset_index(drop=True)

    # ...write back to pickle
    if SAVE:
        df_chains.to_pickle(DATAPATH.joinpath(OP_FILENAME))

    chain_time.stop()

    return df_chains


# . margins
def get_margins(
    MARKET: str,
    cos: list,
    msg: str = "margins",
    RUN_ON_PAPER: bool = False,
    CONCURRENT: int = 200,
    TIMEOUT: int = 5.5,
    FILL_DELAY: int = 5.5,
    OP_FILENAME: str = "",
) -> pd.DataFrame:

    margin_time = Timer(MARKET.lower() + f" {msg}")
    margin_time.start()

    # ... set parameters from var.yml
    ibp = Vars(MARKET.upper())

    HOST, CID = ibp.HOST, ibp.CID
    if RUN_ON_PAPER:
        PORT = ibp.PAPER
    else:
        PORT = ibp.PORT

    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    with IB().connect(HOST, PORT, CID) as ib:
        df = ib.run(
            executeAsync(
                ib=ib,
                algo=margin,
                cts=cos,
                CONCURRENT=CONCURRENT,
                TIMEOUT=TIMEOUT,
                post_process=post_df,
                DATAPATH=DATAPATH,
                OP_FILENAME=OP_FILENAME,
                **{"FILL_DELAY": FILL_DELAY},
            ))

    margin_time.stop()
    return df


# . get contract prices
def get_prices(
    cts: Union[set, list, pd.Series],
    MARKET: str = "",
    RUN_ON_PAPER: bool = False,
    FILL_DELAY: int = 8,
):
    """Gets price of list/set/series of contracts"""

    # ... set parameters from var.yml
    ibp = Vars(MARKET.upper())

    HOST, CID = ibp.HOST, ibp.CID
    if RUN_ON_PAPER:
        PORT = ibp.PAPER
    else:
        PORT = ibp.PORT

    with IB().connect(HOST, PORT, CID) as ib:
        df = ib.run(
            executeAsync(
                ib=ib,
                algo=price,
                cts=cts,
                CONCURRENT=40,
                TIMEOUT=8,
                post_process=post_df,
                **{"FILL_DELAY": FILL_DELAY},
            ))

    return df


# . synchronously qualify


def qualify_opts(
    MARKET: str,
    BLK_SIZE: int = 200,
    RUN_ON_PAPER: bool = True,
    REUSE: bool = True,
    CHECKPOINT: bool = True,
    USE_YAML_DTE: bool = True,
    OP_FILENAME: str = "qopts.pkl",
) -> set:

    # ... start the timer
    opts_time = Timer("qualify options")
    opts_time.start()

    REJECT_FILE = OP_FILENAME[:4] + "_rejects.pkl"
    WIP_FILE = OP_FILENAME[:4] + "_wip.pkl"

    ibp = Vars(MARKET.upper())  # IB Parameters from var.yml

    HOST, CID = ibp.HOST, ibp.CID

    if RUN_ON_PAPER:
        print(f"\nQualifying {MARKET} options using Paper account\n")
        PORT = ibp.PAPER
    else:
        PORT = ibp.PORT

    LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")
    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    # * SETUP LOGS AND CLEAR THEM
    LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_qualsync.log")
    util.logToFile(path=LOGFILE, level=30)
    with open(LOGFILE, "w"):
        pass

    # * LOAD THE FILES
    df_chains = pd.read_pickle(DATAPATH.joinpath("df_chains.pkl"))

    if USE_YAML_DTE:
        df_ch_a = pd.concat(
            (df_chains.assign(right="P"), df_chains.assign(right="C")),
            ignore_index=True,
        )

        # ... remove expired options
        df_ch_a = df_ch_a[df_ch_a.dte > 0].reset_index(drop=True)

        # ... options between min and max dte
        df_ch_b = df_ch_a[df_ch_a.dte.between(ibp.MINDTE,
                                              ibp.MAXDTE,
                                              inclusive=True)]

        # ... options at a dte beyond 6 months for SNP defends
        if MARKET == "SNP":
            defend_dte_dict = (df_ch_a[df_ch_a.dte > eval(ibp.DEFEND_DTE)].
                               groupby("symbol").dte.apply(min).to_dict())

            df_ch_c = df_ch_a[df_ch_a.dte == df_ch_a.symbol.map(
                defend_dte_dict)]

            # ... consolidate
            df_ch = pd.concat([df_ch_b, df_ch_c], ignore_index=True)

        else:
            df_ch = df_ch_b

    # * CLEAN UP THE OPTION CHAIN

    cols = ["symbol", "expiry", "strike", "right"]

    try:
        qopts = pd.read_pickle(DATAPATH.joinpath(OP_FILENAME))
    except FileNotFoundError:
        # for existing successful options
        qopts = pd.Series([], dtype=object, name="qualified")

    try:
        qropts = pd.read_pickle(DATAPATH.joinpath(REJECT_FILE))

        # Clean rejects
        # ... build df_rejects
        df_rejects = (util.df(qropts.to_list()).iloc[:, :6].rename(
            columns={
                "lastTradeDateOrContractMonth": "expiry"
            }).drop(["conId", "secType"], 1).assign(contract=qropts))

        # ... remove df_rejects not in df_ch
        m = df_rejects[cols].apply(tuple, 1).isin(df_ch[cols].apply(tuple, 1))

        qropts = df_rejects[m].contract.rename("rejected")

    except FileNotFoundError:
        # for rejected options
        qropts = pd.Series([], dtype=object, name="rejected")

    try:
        wips = pd.read_pickle(DATAPATH.joinpath(WIP_FILE))
    except FileNotFoundError:
        wips = pd.Series([], dtype=object, name="wip")

    existing_opts = pd.concat([qopts, qropts, wips], ignore_index=True)

    if REUSE:  # remove existing options

        if not existing_opts.empty:  # something is existing!

            df_existing = (util.df(existing_opts.to_list()).iloc[:, :6].rename(
                columns={
                    "lastTradeDateOrContractMonth": "expiry"
                }).drop("conId", 1))

            # remove existing options from df_ch
            df_ch = (pd.concat([df_ch[cols], df_existing[cols]],
                               ignore_index=True).drop_duplicates(
                                   keep=False).reset_index(drop=True))

    # * BUILD THE OPTIONS TO BE QUALIFIED
    cts = [
        Option(s, e, k, r, x) for s, e, k, r, x in zip(
            df_ch.symbol,
            df_ch.expiry,
            df_ch.strike,
            df_ch.right,
            ["NSE" if MARKET.upper() == "NSE" else "SMART"] * len(df_ch),
        )
    ]

    # ..build the raw blocks from cts
    raw_blks = [cts[i:i + BLK_SIZE] for i in range(0, len(cts), BLK_SIZE)]

    with IB().connect(HOST, PORT, CID) as ib:

        for b in tqdm(raw_blks,
                      desc=f"{MARKET} opts qual:",
                      bar_format=BAR_FORMAT,
                      ncols=80):

            qs = ib.run(
                executeAsync(
                    ib=ib,
                    algo=qualify,
                    cts=b,
                    CONCURRENT=200,
                    TIMEOUT=5,
                    post_process=post_df,
                    SHOW_TQDM=True,
                ))

            # Successes
            qopts = qopts.append(pd.Series(qs, dtype=object, name="qualified"),
                                 ignore_index=True)

            # Rejects
            rejects = [c for c in b if not c.conId]
            qropts = qropts.append(pd.Series(rejects,
                                             dtype=object,
                                             name="rejected"),
                                   ignore_index=True)

            if CHECKPOINT:  # store the intermediate options while qualifying
                qopts.to_pickle(DATAPATH.joinpath(WIP_FILE))
                qropts.to_pickle(DATAPATH.joinpath(REJECT_FILE))

        # to prevent TimeoutError()
        ib.disconnect()
        IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

    if CHECKPOINT:
        # ... write final options and cleanup WIP
        qopts.to_pickle(DATAPATH.joinpath(OP_FILENAME))

        try:
            os.remove(DATAPATH.joinpath(WIP_FILE))
        except FileNotFoundError:
            pass

    opts_time.stop()

    return qopts


# .getting opt prices
def opt_prices(
    MARKET: str,
    RUN_ON_PAPER: bool = True,  # Use PAPER account
    OP_FILENAME: str = "df_opt_prices.pkl",
    REUSE: bool = True,
) -> pd.DataFrame:

    # * SETUP

    ibp = Vars(MARKET.upper())  # IB Parameters from var.yml

    opt_price_time = Timer(f"{MARKET} option price")
    opt_price_time.start()

    TEMPL_PATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "template",
                                             "df_price.pkl")
    df_empty = pd.read_pickle(TEMPL_PATH)

    HOST, CID = ibp.HOST, ibp.CID

    if RUN_ON_PAPER:
        print(f"\nGetting prices for {MARKET} options using Paper account\n")
        PORT = ibp.PAPER
    else:
        print(f"\nGetting prices for {MARKET} options using Live account\n")
        PORT = ibp.PORT

    LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")
    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    # * SETUP LOGS AND CLEAR THEM
    LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_price.log")
    util.logToFile(path=LOGFILE, level=30)
    with open(LOGFILE, "w"):
        pass

    # * BUILD UPON EXISTING CONTRACTS
    COLS = ["symbol", "strike", "right", "expiry"]

    # .load existing prices
    if REUSE:

        try:
            df_opt1 = pd.read_pickle(DATAPATH.joinpath(OP_FILENAME))
        except FileNotFoundError:
            df_opt1 = df_empty

        try:
            df_opt2 = pd.read_pickle(DATAPATH.joinpath("z_temp_" +
                                                       OP_FILENAME))
        except FileNotFoundError:
            df_opt2 = df_empty

        df_opt3 = df_opt1.append(df_opt2)

    else:
        df_opt3 = df_empty

        try:
            os.remove(DATAPATH.joinpath("z_temp_" + OP_FILENAME))
        except FileNotFoundError:
            pass

    # . cleanup duplicates
    df_opt3 = df_opt3.drop_duplicates(COLS).reset_index(drop=True)

    # . load ALL option contracts
    qopts = pd.read_pickle(DATAPATH.joinpath("qopts.pkl"))

    # ... convert it to df
    df_q_opts = (util.df(qopts.to_list()).assign(contract=qopts).rename(
        columns={"lastTradeDateOrContractMonth": "expiry"}))

    # ... remove existing price df from df_q_opts

    if df_opt3.empty:
        df_opt4 = df_q_opts

    else:
        m = ~df_q_opts[COLS].apply(tuple, 1).isin(df_opt3[COLS].apply(
            tuple, 1))
        df_opt4 = df_q_opts[m]

        # ... pickle rest in temp for the next time
        df_opt3.to_pickle(DATAPATH.joinpath("z_temp_" + OP_FILENAME))

        if df_opt4.empty:

            print(
                f"\nERROR: All prices are already available in {OP_FILENAME}")
            REGENERATE = yes_or_no(
                f"...Do you want to regenerate ALL prices, ignoring {OP_FILENAME}? "
            )

            if REGENERATE:
                df_opt4 = df_q_opts

                # reset temp for next time
                df_empty.to_pickle(DATAPATH.joinpath("z_temp_" + OP_FILENAME))

            else:
                return df_empty  # !!! Aborted return

    price_contracts = df_opt4.contract

    # * GET THE PRICE AND IV

    with IB().connect(HOST, PORT, CID) as ib:
        df_opt_prices = ib.run(
            executeAsync(
                ib=ib,
                algo=price,
                cts=price_contracts,
                post_process=post_df,
                CONCURRENT=40 * 4,
                TIMEOUT=11,
                DATAPATH=DATAPATH,
                REUSE=True,
                OP_FILENAME="df_opt_prices.pkl",
                **{"FILL_DELAY": 11},
            ))

    # remove NaN from df_opt_prices and save
    # df_opt_prices = df_opt_prices[~df_opt_prices.price.isnull()]\
    #     .reset_index(drop=True)

    opt_price_time.stop()

    return df_opt_prices


# . getting opt margins
def opt_margins(
    MARKET: str,
    RUN_ON_PAPER: bool = True,  # Use PAPER account
    OP_FILENAME: str = "df_opt_margins.pkl",
    REUSE: bool = True,
) -> pd.DataFrame:

    # * SETUP
    opt_margin_time = Timer(f"{MARKET} option margins")
    opt_margin_time.start()

    TEMPL_PATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "template",
                                             "df_margin.pkl")
    df_empty = pd.read_pickle(TEMPL_PATH)
    OUTPUT_COLS = list(
        df_empty.drop(["commission", "maxCommission", "initMarginChange"],
                      1).columns)

    LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")
    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    # * SETUP LOGS AND CLEAR THEM
    LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_margin.log")
    util.logToFile(path=LOGFILE, level=30)
    with open(LOGFILE, "w"):
        pass

    # * BUILD UPON EXISTING CONTRACTS
    COLS = ["symbol", "strike", "right", "expiry"]

    # .load existing margins
    if REUSE:

        try:
            df_opt1 = pd.read_pickle(DATAPATH.joinpath(OP_FILENAME))
        except FileNotFoundError:
            df_opt1 = df_empty[OUTPUT_COLS]

        try:
            df_opt2 = pd.read_pickle(DATAPATH.joinpath("z_temp_" +
                                                       OP_FILENAME))
        except FileNotFoundError:
            df_opt2 = df_empty[OUTPUT_COLS]

        df_opt3 = df_opt1.append(df_opt2)
        df_opt3 = df_opt3[OUTPUT_COLS]

    else:
        df_opt3 = df_empty[OUTPUT_COLS]

        try:
            os.remove(DATAPATH.joinpath("z_temp_" + OP_FILENAME))
        except FileNotFoundError:
            pass

    # .cleanup duplicates
    df_opt3 = df_opt3.drop_duplicates(COLS).reset_index(drop=True)

    # . load ALL option contracts
    qopts = pd.read_pickle(DATAPATH.joinpath("qopts.pkl"))

    # ... convert it to df
    df_q_opts = (util.df(qopts.to_list()).assign(contract=qopts).rename(
        columns={"lastTradeDateOrContractMonth": "expiry"}))

    # ... integrate lots (from chains)

    df_chains = pd.read_pickle(
        DATAPATH.joinpath("df_chains.pkl")).drop_duplicates()

    col1 = ["symbol", "strike", "expiry"]
    df_q_opts = (df_q_opts.set_index(col1).join(
        df_chains.set_index(col1)[["lot"]]).reset_index())

    # ... process dtes
    df_q_opts["dte"] = df_q_opts.expiry.apply(get_dte)
    df_q_opts = df_q_opts[df_q_opts.dte > 0]  # Remove negative dtes

    # Make 0 dte positive to avoid sqrt errors
    df_q_opts.loc[df_q_opts.dte == 0, "dte"] = 1

    # ... remove existing margin df from df_q_opts

    if df_opt3.empty:
        df_opt4 = df_q_opts

    else:
        m = ~df_q_opts[COLS].apply(tuple, 1).isin(df_opt3[COLS].apply(
            tuple, 1))
        df_opt4 = df_q_opts[m]

        # ... pickle rest in temp for the next time
        df_opt3.to_pickle(DATAPATH.joinpath("z_temp_" + OP_FILENAME))

        if df_opt4.empty:

            print(
                f"\nERROR: All margins are already available in {OP_FILENAME}")
            REGENERATE = yes_or_no(
                f"...Do you want to regenerate ALL margins, ignoring {OP_FILENAME}? "
            )

            if REGENERATE:
                df_opt4 = df_q_opts

                # reset temp for next time
                df_empty[OUTPUT_COLS].to_pickle(
                    DATAPATH.joinpath("z_temp_" + OP_FILENAME))

            else:
                return df_empty[OUTPUT_COLS]  # !!! Aborted return

    # ... remove any duplicate conIds
    df_opt4 = df_opt4.drop_duplicates(["conId"]).reset_index(drop=True)

    mgn_contracts = df_opt4.contract

    mgn_orders = [
        MarketOrder("SELL", lot / lot)
        if MARKET.upper() == "SNP" else MarketOrder("SELL", lot)
        for lot in df_opt4.lot
    ]

    opt_cos = [(c, o) for c, o in zip(mgn_contracts, mgn_orders)]

    # * GET THE MARGINS
    df_opt_margins = get_margins(
        MARKET=MARKET,
        cos=opt_cos,
        msg="option margins",
        RUN_ON_PAPER=RUN_ON_PAPER,
        CONCURRENT=200,
        TIMEOUT=5.5,
        FILL_DELAY=5.5,
        OP_FILENAME="df_opt_margins.pkl",
    )

    # remove NaN from margins
    df_opt_margins = df_opt_margins[~df_opt_margins.margin.isnull(
    )].reset_index(drop=True)
    df_opt_margins = df_opt_margins[OUTPUT_COLS]

    opt_margin_time.stop()

    return df_opt_margins
