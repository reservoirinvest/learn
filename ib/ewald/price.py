# Get historical price data of a ticker

import asyncio
import datetime
import pathlib
import time
from collections import defaultdict
from tqdm import tqdm

import pandas as pd
import numpy as np
from ib_insync import IB, util

SYMBOL = 'PNB'
MARKET = 'NSE'

BAR_FORMAT = "{desc:<10}{percentage:3.0f}%|{bar:25}{r_bar}{bar:-10b}"

start_time = time.time()

# Connect
ib = IB()
ib.connect('127.0.0.1', 3001, clientId=4)

# Prepare contracts
datapath = pathlib.Path.cwd().joinpath('data', MARKET.lower())
df_opts = pd.read_pickle(datapath.joinpath('df_opts.pkl'))
df_unds = pd.read_pickle(datapath.joinpath('df_unds.pkl'))

# Nearest dte
df_sym = df_opts[df_opts.symbol == SYMBOL] # filter symbol
df_dte = df_sym[df_sym.dte == df_sym.dte.unique().min()] # filter dte

# sort from strike nearest to undPrice to farthest
df_dte = df_dte.iloc[abs(df_dte.strike-df_dte.undPrice.iloc[0]).argsort()]
contracts = df_dte.contract.unique()

print(f"\nLength of contracts is: {len(contracts)}\n") # !!! TEMPORARY
contracts = contracts[:5] # !!! DATA LIMITER

# contract = contracts[50] # a contract at the edge

start = ''
end = datetime.datetime.now()

# ticks = ib.reqHistoricalTicks(contract, start, end, 1, 'BID_ASK', useRth=False, ignoreSize=False)
# print({contract.localSymbol: ticks})

async def hist_async(ib, contracts):
    start = ''
    end = datetime.datetime.utcnow()

    bid_asks = [ib.reqHistoricalTicksAsync(c, start, end, 1, 'BID_ASK', useRth=False, ignoreSize=False) for c in contracts]
    trades = [ib.reqHistoricalTicksAsync(c, start, end, 1, 'TRADES', useRth=False, ignoreSize=False) for c in contracts]

    # ba = await asyncio.gather(*bid_asks)
    # t = await asyncio.gather(*trades)

    ba = [await f for f in tqdm(asyncio.as_completed(bid_asks), 
                            desc='bid_ask:', total=len(bid_asks), 
                            bar_format=BAR_FORMAT)]

    t = [await f for f in tqdm(asyncio.as_completed(trades), 
                            desc='last:', total=len(trades), 
                            bar_format=BAR_FORMAT)]

    ba_d = defaultdict(dict) # dict of attributes dicts for bid-ask
    t_d = defaultdict(dict) # dict of attributes dicts for trades
    
    for i, v in enumerate(ba):

        tick = v[-1] # take the last tick

        a = defaultdict(dict) # attribute dict
        a['ba_time'] = tick.time
        a['bid'] = tick.priceBid
        a['ask'] = tick.priceAsk

        ba_d[contracts[i].conId] = a

    for i, v in enumerate(t):

        tick = v[-1] # take the last tick

        a = defaultdict(dict) # attribute dict
        a['t_time'] = tick.time
        a['last'] = tick.price
        
        t_d[contracts[i].conId] = a

    df = util.df(contracts.tolist()).iloc[:, :6]\
             .rename({'lastTradeDateOrContractMonth': 'expiry'})\
                 .set_index('conId')\
                     .join(pd.DataFrame(ba_d).T.join(pd.DataFrame(t_d).T))

    # print(df)

    # result = {[c.conId for c in contracts], ticks}

    return df

df = ib.run(hist_async(ib, contracts))

print(df)

print(f"\n... took {time.time()-start_time:.2f} seconds")
