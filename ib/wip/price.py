# Experiment with prices - based on `price.ipynb` notebook

import asyncio
import pathlib
import sys
import time
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from ib_insync import IB, util
from tqdm import tqdm

MARKET = 'SNP'

# Set root path to parent
this_file_path = pathlib.Path(__file__).parent.absolute()
root = pathlib.Path(this_file_path.parent)
sys.path.append(root)



# Build dfs
datapath = root.joinpath(root, 'data',  MARKET.lower())
df_symlots = pd.read_pickle(datapath.joinpath('df_symlots.pkl'))
qopts = pd.read_pickle(datapath.joinpath('qopts.pkl'))

# Stage contracts
# contracts = df_symlots.contract.unique()[:2] # !!! DATA LIMITER
# contracts_for_async = df_symlots.contract.unique()
contracts_for_async = qopts.sample(3).to_list()  # !!! DATA LIMITER
contract = df_symlots.contract.sample(1).iloc[0]

HOST = '127.0.0.1'
PORT = 1301

## ** SIMPLE ONE-CONTRACT PRICE **

def mkt_ticks(contracts):
    ib = IB().connect(HOST, PORT, 0)
    [ib.reqMktData(c) for c in contracts]
    ticks = [ib.reqTickers(c) for c in contracts]
    [ib.cancelMktData(c) for c in contracts]

    return ticks

## ** MULTI-CONTRACT PRICE USING MARKET DATA **
# Note that it is faster to use execAsync with price instead of this.

async def mkt_ticksAsync(contracts):
    
    with await IB().connectAsync(HOST, PORT, 0) as ib:
        ib.client.setConnectOptions('+PACEAPI')
        ticks = await asyncio.gather(*[ib.reqTickersAsync(c) for c in contracts])

    df = util.df([i for t in ticks for i in t]).dropna(subset=['time'])

    return df

## ** MULTI-CONTRACT PRICE FROM HISTORY **
# Uses `bid`, `ask` and `trade` from historical data bars

async def ba_async(ib, contracts):
   
    async def coro(c):
        
        lastPrice = await ib.reqHistoricalDataAsync(
                            contract=c,
                            endDateTime='',
                            durationStr='30 S',
                            barSizeSetting='30 secs',
                            whatToShow='TRADES',
                            useRTH=False,
                            formatDate=2,
                            keepUpToDate=False,
                            timeout=0)
        
        bidPrice = await ib.reqHistoricalDataAsync(
                            contract=c,
                            endDateTime='',
                            durationStr='30 S',
                            barSizeSetting='30 secs',
                            whatToShow='BID',
                            useRTH=False,
                            formatDate=2,
                            keepUpToDate=False,
                            timeout=0)
        
        askPrice = await ib.reqHistoricalDataAsync(
                            contract=c,
                            endDateTime='',
                            durationStr='30 S',
                            barSizeSetting='30 secs',
                            whatToShow='ASK',
                            useRTH=False,
                            formatDate=2,
                            keepUpToDate=False,
                            timeout=0)
        
        try:
            date = lastPrice[-1].date
            last = lastPrice[-1].close

        except IndexError:
            date = pd.NaT
            last = np.nan
            
        try:
            bid = bidPrice[-1].close
        except IndexError:
            bid = np.nan
            
        try:
            ask = askPrice[-1].close
        except IndexError:
            ask = np.nan
            
        return {c.conId: {'date': date, 'last': last, "bid": bid, "ask": ask}}
    
    tsks= [asyncio.create_task(coro(c), name=c.conId) for c in contracts]
    tasks = [await f for f in tqdm(asyncio.as_completed(tsks), total=len(tsks))]
    
    d = defaultdict(dict)
    for t in tasks:
        for k, v in t.items():
            d[k]=v
            
    df = pd.DataFrame(d).T
    
    return df

start_time = time.time()

# print(mkt_ticks(contracts))
# print(asyncio.run(mkt_ticksAsync(contracts_for_async)))

with IB().connect(HOST, PORT, 0) as ib:
    ib.client.setConnectOptions('+PACEAPI')
    print(asyncio.run(ba_async(ib, contracts_for_async)))

print(f"\n... took {time.time()-start_time:.2f} seconds")

