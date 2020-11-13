# Common support functions

import asyncio
import datetime
import math
import os
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from ib_insync import IB, Stock, util
from pytz import timezone

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
VAR_YML = os.path.join(THIS_FOLDER, "var.yml")


# ** CLASSES
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


def get_dte(dt):
    """Gets days to expiry

    Arg:
        (dt) as day in string format 'yyyymmdd'
    Returns:
        days to expiry as int"""

    try:
        dte = (util.parseIBDatetime(dt) -
               datetime.datetime.utcnow().date()).days
    except Exception:
        dte = None

    return dte


def calcsdmult_df(price, df):
    '''Back calculate standard deviation MULTIPLE against undPrice for given price. Needs dataframes.

    Args:
        (price) as series of price whose sd needs to be known in float
        (df) as a dataframe with undPrice, dte and iv columns in float

    Returns:
        Series of std deviation multiple as float

        '''
    sdevmult = (price - df.undPrice) / \
        ((df.dte / 365).apply(math.sqrt) * df.iv * df.undPrice)
    return abs(sdevmult)


def calcsd(price, undPrice, dte, iv):
    '''Calculate standard deviation MULTIPLE for given price.

    Args:
        (price) the price whose sd needs to be known in float
        (undPrice) the underlying price in float
        (dte) the number of days to expiry in int
        (iv) the implied volatility in float

    Returns:
        Std deviation of the price in float

        '''
    try:
        sdev = abs((price - undPrice) / (math.sqrt(dte / 365) * iv * undPrice))
    except Exception:
        sdev = np.nan
    return sdev


def abs_int(x):
    '''Makes a proper int or give out a none. Used to prevent divide-by-zero for pandas.

    Arg:
        x as the input with inf/-inf/nan
    Returns:
        abs(int) | None

    '''
    try:
        y = abs(int(x))
    except Exception:
        return None
    return y


async def isMarketOpen(ib: IB, MARKET: str) -> bool:
    '''Determines if market is open or not

    Args:

        (ib): as connection object,
        (MARKET): ('NSE'|'SNP')

    Returns:
        bool

    Note: 
     - Though IB uses UTC elsewhere, in contract details `zone` is available as a string!
     - ...hence times are converted to local market times for comparison

    '''

    # Establish the timezones
    tzones = {'UTC': timezone('UTC'),
              'IST': timezone('Asia/Kolkata'),
              'EST': timezone('US/Eastern')}

    if MARKET.upper() == 'NSE':
        ct = await ib.qualifyContractsAsync(
            Stock(symbol='RELIANCE', exchange='NSE', currency='INR'))
    elif MARKET.upper() == 'SNP':
        ct = await ib.qualifyContractsAsync(
            Stock(symbol='INTC', exchange='SMART', currency='USD'))
    else:
        print(f'\nUnknown market {MARKET}!!!\n')
        return None

    ctd = await ib.reqContractDetailsAsync(ct[0])

    hrs = util.df(ctd).liquidHours.iloc[0].split(';')
    zone = util.df(ctd).timeZoneId.iloc[0].split(' ')[0]

    # Build the time dataframe by splitting hrs
    tframe = pd.DataFrame([re.split(':|-|,', h) for h in hrs]).\
        rename(columns={0: 'from', 1: 'start', 2: 'to', 3: 'end'})
    tframe['zone'] = zone
    tframe['now'] = datetime.now(tzones[zone])

    tframe = tframe.dropna()

    open = pd.to_datetime(tframe['from'] + tframe['start']).\
        apply(lambda x: x.replace(tzinfo=tzones[zone]))
    close = pd.to_datetime(tframe['to'] + tframe['end']).\
        apply(lambda x: x.replace(tzinfo=tzones[zone]))

    tframe = tframe.assign(open=open, close=close)
    tframe = tframe.assign(isopen=(tframe['now'] >= tframe['open']) &
                           (tframe['now'] <= tframe['close']))

    market_open = any(tframe['isopen'])

    return market_open


def quick_pf(ib) -> pd.DataFrame:

    pf = ib.portfolio()   # returns an empty [] if there is nothing in the portfolio

    if pf != []:
        df_pf = util.df(pf)
        df_pf = (util.df(list(df_pf.contract)).iloc[:, :6]).join(
            df_pf.drop(['contract', 'account'], 1))
        df_pf = df_pf.rename(columns={'lastTradeDateOrContractMonth': 'expiry',
                                      'marketPrice': 'mktPrice',
                                      'marketValue': 'mktVal',
                                      'averageCost': 'avgCost',
                                      'unrealizedPNL': 'unPnL',
                                      'realizedPNL': 'rePnL'})
    else:
        df_pf = pd.read_pickle('./data/template/df_portfolio.pkl')

    return df_pf


async def get_pnl(ib, MARKET):

    with open(VAR_YML) as f:
        data = yaml.safe_load(f)

    POSN_LIMIT = data[MARKET.upper()]["POSN_LIMIT"]
    CUSHION_LIMIT = data[MARKET.upper()]["CUSHION_LIMIT"]

    MAX_SLEEP = 2  # keep this below 2 seconds
    MIN_SLEEP = 0.01
    CUM_SLEEP = 0

    acct = ib.managedAccounts()[0]
    pnlobj = ib.reqPnL(acct)

    df_mgns = util.df(ib.accountValues())
    df_mgns = df_mgns[df_mgns.tag.isin(['NetLiquidation', 'InitMarginReq',
                                        'EquityWithLoanValue', 'MaintMarginReq', 'AccruedCash'])]
    margins = df_mgns.set_index('tag').value.apply(float).to_dict()

    # sleep to populate PnL object
    while np.isnan(pnlobj.dailyPnL) & (CUM_SLEEP < MAX_SLEEP):
        await asyncio.create_task(asyncio.sleep(MIN_SLEEP))
        CUM_SLEEP = CUM_SLEEP + MIN_SLEEP
        await asyncio.create_task(asyncio.sleep(0.01))

    # await asyncio.sleep(1)  # To fill pnl data

    # assemble the PnL
    pnl = defaultdict(dict)
    pnl['NLV'] = margins['NetLiquidation']
    pnl['dailyPnL'] = pnlobj.dailyPnL
    pnl['unPnL'] = pnlobj.unrealizedPnL
    pnl['rePnL'] = pnlobj.realizedPnL
    pnl['initialMargin'] = margins['InitMarginReq']
    pnl['maintMargin'] = margins['MaintMarginReq']
    pnl['mtdInterest'] = margins['AccruedCash']
    pnl['cushion'] = (margins['EquityWithLoanValue'] -
                      margins['MaintMarginReq']) / margins['NetLiquidation']
    pnl['cushionLmt'] = CUSHION_LIMIT
    pnl['perPositionLmt'] = margins['NetLiquidation'] * POSN_LIMIT

    return pnl

if __name__ == "__main__":
    print(get_dte("20210101"))
