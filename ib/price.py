# testing prices (again!!)
import asyncio
import pathlib
from collections import defaultdict, namedtuple
from datetime import datetime

import numpy as np
import pandas as pd
from ib_insync import IB, Contract, util
from tqdm import tqdm

from engine import Vars

MARKET = 'NSE'

ibp = Vars(MARKET.upper())

# set and empty log file
logf = pathlib.Path.cwd().joinpath('data', 'log', 'temp.log')
util.logToFile(path=logf, level=30)
with open(logf, "w"):
    pass

datapath = pathlib.Path.cwd().joinpath('data', MARKET.lower())
df_nakeds = pd.read_pickle(datapath.joinpath('df_nakeds.pkl'))

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

    print(f"\ntick:{tick}") # !!! TEMPORARY

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

    task = asyncio.wait_for(price(ib, contract, **kwargs), timeout=FILL_DELAY)

    res = await asyncio.gather(task)

    print(f"\nresult: {res}\n") # !!! TEMPORARY

    try:
        df_mktpr = res[0]
    except IndexError:
        df_mktpr = pd.DataFrame([])
    
    # df_mktpr = await price(ib, contract, **kwargs)
    
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
                            ignoreSize=False), timeout=FILL_DELAY)

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
    
    timeout=15
    tasks = []

    if hasattr(contracts, '__iter__'):
        tasks = [qpCoro(ib=ib, contract=contract, **kwargs) for contract in contracts]
    else:
        for contract in contracts:
            task = asyncio.wait_for(qpCoro(ib=ib, contract=contract, **kwargs), timeout)
            tasks.append(task)

    df_prs = [await res for res in tqdm(asyncio.as_completed(tasks), total=len(tasks))]
        
    df = pd.concat(df_prs, ignore_index=True)
    return df

with IB().connect(ibp.HOST, ibp.PORT, ibp.CID) as ib:
    df_pr = ib.run(qpAsync(ib, contracts=df_nakeds.contract[-2:], **{'FILL_DELAY': 5.5}))

print(df_pr)
