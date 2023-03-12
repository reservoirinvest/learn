# !!! TEMPORARY PROGRAM AS ORDER PLACEMENT WAS NOT WORKING ON 24-MAR-2021

MARKET = 'NSE'
import sys
import pathlib
import numpy as np
import pandas as pd

from ib_insync import IB, util, Option, MarketOrder, Contract
from typing import Callable, Coroutine, Union

# Get capability to import programs from `asyncib` folder
cwd = pathlib.Path.cwd() # working directory from where python was initiated
DATAPATH = cwd.joinpath('data', MARKET.lower()) # path to store data files
LOGFILE = DATAPATH.joinpath('temp.log') # path to store log files

IBPATH = cwd.parent.parent.joinpath('asyncib') # where ib programs are stored

# append IBPATH to import programs.
if str(IBPATH) not in sys.path:  # Convert it to string!
    sys.path.append(str(IBPATH))
    
IBDATAPATH = IBPATH.joinpath('data', MARKET.lower())

# Get the host, port, cid
from engine import Vars

ibp = Vars(MARKET.upper())  # IB Parameters from var.yml
HOST, PORT, CID = ibp.HOST, ibp.PORT, ibp.CID

# Get the pickle files
from os import listdir
fs = listdir(DATAPATH)

files = [f for f in fs if f[-4:] == '.pkl']
for f in files:
    exec(f"{f.split('.')[0]} = pd.read_pickle(DATAPATH.joinpath(f))")
np.sort(np.array(files))

# * IMPORTS
from ib_insync import LimitOrder
from support import get_openorders, place_orders

# * RUN nakeds
THIS_FOLDER = ''

df_nakeds = pd.read_pickle(DATAPATH.joinpath(THIS_FOLDER, 'df_nakeds.pkl'))

cols = ['symbol', 'strike', 'right', 'expiry', 'dte', 'conId', 'contract',
        'margin', 'bid', 'ask', 'iv', 'und_iv','price', 'lot', 'rom', 'sdMult', 'expRom', 'expPrice', 'qty']

df = df_nakeds[df_nakeds.price>0][cols].sort_values('rom', ascending=False)

# * REMOVE OPEN ORDERS FROM NAKEDS df
df_openords = get_openorders(MARKET)
df = df[~df.conId.isin(df_openords.conId)].reset_index(drop=True)

# ... build the naked SELL orders
contracts = df.contract.to_list()
orders = [LimitOrder(action='SELL', totalQuantity=abs(int(q)), lmtPrice=p) 
                        for q, p in zip(df.qty, df.expPrice)]

naked_cos = tuple((c, o) for c, o in zip(contracts, orders))

with IB().connect(HOST, PORT, CID) as ib:
    ordered = place_orders(ib=ib, cos=naked_cos)

# print(df)