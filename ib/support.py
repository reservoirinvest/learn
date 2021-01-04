# Common support functions

import asyncio
import datetime
import math
import os
import pathlib
import re
import time
from collections import defaultdict, namedtuple
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import ta
import yaml
from ib_insync import IB, Contract, Stock, util
from pytz import timezone
from scipy.integrate import quad
from tqdm import tqdm

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

        print(
            f'\n{self.name} started at {time.strftime("%d-%b-%Y %H:%M:%S", time.localtime())}'
        )

        self._start_time = time.perf_counter()

    def stop(self) -> None:
        if self._start_time is None:
            raise Exception(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time

        print(
            f"\n...{self.name} took: " +
            f"{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} seconds\n"
        )

        self._start_time = None


def abs_int(x):
    """Makes a proper int or give out a none. Used to prevent divide-by-zero for pandas.

    Arg:
        x as the input with inf/-inf/nan
    Returns:
        abs(int) | None

    """
    try:
        y = abs(int(x))
    except Exception:
        return None
    return y


def calcsdmult_df(price, df):
    """Back calculate standard deviation MULTIPLE against undPrice for given price. Needs dataframes.

    Args:
        (price) as series of price whose sd needs to be known in float
        (df) as a dataframe with undPrice, dte and iv columns in float

    Returns:
        Series of std deviation multiple as float

    """
    sdevmult = (price - df.undPrice) / (
        (df.dte / 365).apply(math.sqrt) * df.iv * df.undPrice)
    return abs(sdevmult)


def calcsd(price, undPrice, dte, iv):
    """Calculate standard deviation MULTIPLE for given price.

    Args:
        (price) the price whose sd needs to be known in float
        (undPrice) the underlying price in float
        (dte) the number of days to expiry in int
        (iv) the implied volatility in float

    Returns:
        Std deviation of the price in float

    """
    try:
        sdev = abs((price - undPrice) / (math.sqrt(dte / 365) * iv * undPrice))
    except Exception:
        sdev = np.nan
    return sdev


def empty_trash(MARKET: str):
    """Delete trash from DATAPATH and LOGPATH"""

    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())
    LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")

    do_not_delete = [
        ".keep",
        "qopts.pkl",
        "qopt_rejects.pkl",
        "df_symlots.pkl",
        "df_unds.pkl",
        "df_chains.pkl",
        "df_ohlcs.pkl",
        "df_opts.pkl"
    ]

    data_fs = os.listdir(DATAPATH)
    data_files = [f for f in data_fs if f not in do_not_delete]

    for f in data_files:
        try:
            os.remove(DATAPATH.joinpath(f))
        except (FileNotFoundError, PermissionError):
            print(f"\n...file {f} could not be deleted!")

    log_fs = os.listdir(LOGPATH)
    log_files = [f for f in log_fs if f[:3] == MARKET.lower()]

    for f in log_files:
        try:
            os.remove(LOGPATH.joinpath(f))
        except (FileNotFoundError, PermissionError):
            pass

    return None


def fallrise(df_hist, dte):
    """Gets the fall and rise for a specific symbol and dte

    Args:
        (df_hist) as a df with historical ohlc for a scrip
        (dte) as int for days to expiry
    Returns:
        symbol named tuple(symbol, dte, fall, rise)"""

    s = df_hist.symbol.unique()[0]
    df = df_hist.set_index("date").sort_index(ascending=True)
    df = df.assign(
        delta=df.high.rolling(dte).max() - df.low.rolling(dte).min(),
        pctchange=df.close.pct_change(periods=dte),
    )

    max_fall = df[df.pctchange <= 0].delta.max()
    max_rise = df[df.pctchange > 0].delta.max()
    
    fallrise = namedtuple(s, ['symbol', 'dte', 'fall', 'rise'])
    res = fallrise(symbol=s, dte=dte, fall=round(max_fall, 2), rise=round(max_rise, 2))

    return res


def get_col_widths(dataframe):
    """Provide column widths for `auto-fitting` pandas dataframe"""

    widths = [
        max([
            len(str(round(s, 2))) if isinstance(s, float) else len(str(s))
            for s in dataframe[col].values
        ] + [len(col) * 1.2]) for col in dataframe.columns
    ]

    return widths


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


def get_market(msg: str = "") -> str:

    # . get user input for market
    mkt_dict = {1: "NSE", 2: "SNP"}
    mkt_ask_range = [i + 1 for i in list(range(len(mkt_dict)))]
    mkt_ask = f"\n{msg}\n"
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

    return MARKET


def get_openorders(MARKET: str) -> pd.DataFrame:
    """Gets openorders

    Arg:
        (MARKET) as str SNP|NSE

    Returns:
        dataframe of openorders

    """

    TMPLTPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "template")

    # initialize yaml variables
    # ... set parameters from var.yml
    ibp = Vars(MARKET.upper())

    HOST, PORT, CID = ibp.HOST, ibp.PORT, ibp.MASTERCID
    ACTIVE_STATUS = ibp.ACTIVE_STATUS

    # .. initialize openorder dataframe from template
    df_openords = pd.read_pickle(TMPLTPATH.joinpath("df_openords.pkl"))

    with IB().connect(HOST, PORT, CID) as ib:
        ib.reqAllOpenOrders()  # To kickstart collection of open orders
        # ib.sleep(0.3)
        trades = ib.trades()

    if trades:
        all_trades_df = (util.df(t.contract for t in trades).join(
            util.df(t.orderStatus
                    for t in trades)).join(util.df(t.order for t in trades),
                                           lsuffix="_"))

        all_trades_df.rename({"lastTradeDateOrContractMonth": "expiry"},
                             axis="columns",
                             inplace=True)
        trades_cols = [
            "conId",
            "symbol",
            "secType",
            "expiry",
            "strike",
            "right",
            "orderId",
            "permId",
            "action",
            "totalQuantity",
            "lmtPrice",
            "status",
        ]

        dfo = all_trades_df[trades_cols]
        df_openords = dfo[all_trades_df.status.isin(ACTIVE_STATUS)]

    return df_openords


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
    df_mgns = df_mgns[df_mgns.tag.isin([
        "NetLiquidation",
        "InitMarginReq",
        "EquityWithLoanValue",
        "MaintMarginReq",
        "AccruedCash",
    ])]
    margins = df_mgns.set_index("tag").value.apply(float).to_dict()

    # sleep to populate PnL object
    while np.isnan(pnlobj.dailyPnL) & (CUM_SLEEP < MAX_SLEEP):
        await asyncio.create_task(asyncio.sleep(MIN_SLEEP))
        CUM_SLEEP = CUM_SLEEP + MIN_SLEEP
        await asyncio.create_task(asyncio.sleep(0.01))

    # await asyncio.sleep(1)  # To fill pnl data

    # assemble the PnL
    pnl = defaultdict(dict)
    pnl["NLV"] = margins["NetLiquidation"]
    pnl["dailyPnL"] = pnlobj.dailyPnL
    pnl["unPnL"] = pnlobj.unrealizedPnL
    pnl["rePnL"] = pnlobj.realizedPnL
    pnl["initialMargin"] = margins["InitMarginReq"]
    pnl["maintMargin"] = margins["MaintMarginReq"]
    pnl["mtdInterest"] = margins["AccruedCash"]
    pnl["cushion"] = (margins["EquityWithLoanValue"] -
                      margins["MaintMarginReq"]) / margins["NetLiquidation"]
    pnl["cushionLmt"] = CUSHION_LIMIT
    pnl["perPositionLmt"] = margins["NetLiquidation"] * POSN_LIMIT

    return pnl


def get_prec(v, base):
    """Gives the precision value

    Args:
       (v) as value needing precision in float
       (base) as the base value e.g. 0.05
    Returns:
        the precise value"""

    try:
        output = round(
            round((v) / base) * base, -int(math.floor(math.log10(base))))
    except Exception:
        output = None

    return output


def get_prob(sd):
    """Compute probability of a normal standard deviation

    Arg:
        (sd) as standard deviation
    Returns:
        probability as a float

    """
    prob = quad(lambda x: np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi), -sd, sd)[0]
    return prob

def get_rsi(df_ohlcs: pd.DataFrame, # df with ascending series of close,
            days: int=14, # no of days for the rsi
           ):
    
    if len(df_ohlcs.symbol.unique()) > 1:
        df = df_ohlcs.set_index('date').sort_index(ascending=True)
        df = df.groupby('symbol').close.apply(lambda x: ta.momentum.RSIIndicator(close=x, window=14).rsi().iloc[-1])
        rsi = df.rename('rsi')
    else:
        rsi = ta.momentum.RSIIndicator(close=df_ohlcs.close,  window=days).rsi().iloc[-1]
        
    return rsi


async def isMarketOpen(ib: IB, MARKET: str) -> bool:
    """Determines if market is open or not

    Args:

        (ib): as connection object,
        (MARKET): ('NSE'|'SNP')

    Returns:
        bool

    Note:
     - Though IB uses UTC elsewhere, in contract details `zone` is available as a string!
     - ...hence times are converted to local market times for comparison

    """

    # Establish the timezones
    tzones = {
        "UTC": timezone("UTC"),
        "IST": timezone("Asia/Kolkata"),
        "EST": timezone("US/Eastern"),
    }

    if MARKET.upper() == "NSE":
        ct = await ib.qualifyContractsAsync(
            Stock(symbol="RELIANCE", exchange="NSE", currency="INR"))
    elif MARKET.upper() == "SNP":
        ct = await ib.qualifyContractsAsync(
            Stock(symbol="INTC", exchange="SMART", currency="USD"))
    else:
        print(f"\nUnknown market {MARKET}!!!\n")
        return None

    ctd = await ib.reqContractDetailsAsync(ct[0])

    hrs = util.df(ctd).liquidHours.iloc[0].split(";")
    zone = util.df(ctd).timeZoneId.iloc[0].split(" ")[0]

    # Build the time dataframe by splitting hrs
    tframe = pd.DataFrame([re.split(":|-|,", h) for h in hrs]).rename(columns={
        0: "from",
        1: "start",
        2: "to",
        3: "end"
    })
    tframe["zone"] = zone
    tframe["now"] = datetime.now(tzones[zone])

    tframe = tframe.dropna()

    open = pd.to_datetime(tframe["from"] + tframe["start"]).apply(
        lambda x: x.replace(tzinfo=tzones[zone]))
    close = pd.to_datetime(tframe["to"] + tframe["end"]).apply(
        lambda x: x.replace(tzinfo=tzones[zone]))

    tframe = tframe.assign(open=open, close=close)
    tframe = tframe.assign(isopen=(tframe["now"] >= tframe["open"])
                           & (tframe["now"] <= tframe["close"]))

    market_open = any(tframe["isopen"])

    return market_open


def place_orders(ib: IB, cos: Union[Tuple, List], blk_size: int = 25) -> List:
    """!!!CAUTION!!!: This places orders in the system
    NOTE: cos could be a single (contract, order)
          or a tuple/list of ((c1, o1), (c2, o2)...)
          made using tuple(zip(cts, ords))"""

    trades = []

    if isinstance(cos, (tuple, list)) and (len(cos) == 2):
        c, o = cos
        trades.append(ib.placeOrder(c, o))

    else:
        cobs = {cos[i:i + blk_size] for i in range(0, len(cos), blk_size)}

        for b in tqdm(cobs):
            for c, o in b:
                td = ib.placeOrder(c, o)
                trades.append(td)
            ib.sleep(0.75)

    return trades


def quick_pf(ib) -> pd.DataFrame:

    pf = ib.portfolio(
    )  # returns an empty [] if there is nothing in the portfolio

    if pf != []:
        df_pf = util.df(pf)
        df_pf = (util.df(list(df_pf.contract)).iloc[:, :6]).join(
            df_pf.drop(["contract", "account"], 1))
        df_pf = df_pf.rename(
            columns={
                "lastTradeDateOrContractMonth": "expiry",
                "marketPrice": "mktPrice",
                "marketValue": "mktVal",
                "averageCost": "avgCost",
                "unrealizedPNL": "unPnL",
                "realizedPNL": "rePnL",
            })
    else:
        TEMPL_PATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data",
                                                 "template")
        df_pf = pd.read_pickle(TEMPL_PATH.joinpath("df_portfolio.pkl"))

    return df_pf


async def quick_price_async(ib: IB, contract: Contract) -> pd.DataFrame:

    # Check for executeAsync engine
    if isinstance(contract, tuple):
        contract = contract[0]

    result = defaultdict(dict)

    ticks = await asyncio.gather(
        ib.reqHistoricalTicksAsync(
            contract=contract,
            startDateTime="",
            endDateTime=datetime.datetime.now(),
            numberOfTicks=1,
            whatToShow="Bid_Ask",
            useRth=False,
            ignoreSize=False,
        ),
        ib.reqHistoricalTicksAsync(
            contract=contract,
            startDateTime="",
            endDateTime=datetime.datetime.now(),
            numberOfTicks=1,
            whatToShow="Trades",
            useRth=False,
            ignoreSize=False,
        ),
    )

    # extract bid and ask price
    try:
        bid_ask = ticks[0][-1]  # bid ask is not availble for Index securities!
        result["bid"] = bid_ask.priceBid
        result["ask"] = bid_ask.priceAsk

    except IndexError:
        print(
            f"\nNo bid-ask for {contract.localSymbol} of secType: {contract.secType}"
        )
        result["bid"] = np.nan
        result["ask"] = np.nan

    # extract last reported price
    try:
        # pick reported price if available
        result["last"] = [
            t.price for t in ticks[1] if not t.tickAttribLast.unreported
        ][-1]
    except IndexError:
        # pick up last tick price

        try:
            result["last"] = ticks[1][-1].price
        except IndexError:
            result["last"] = np.nan

    # . build the df
    df_pr = pd.DataFrame([
        pd.Series(contract.conId, name="conId"),
        pd.Series(contract.symbol, name="symbol"),
        pd.Series(contract.localSymbol, name="localSymbol"),
        pd.Series(result["bid"], name="bid", dtype="float64"),
        pd.Series(result["ask"], name="ask", dtype="float64"),
        pd.Series(result["last"], name="last", dtype="float64"),
    ]).T

    # . use bid-ask avg if last price is not available
    df_pr = df_pr.assign(
        price=df_pr["last"].combine_first(df_pr[["bid", "ask"]].mean(axis=1)))

    return df_pr


def watchlist(MARKET: str,
              symbols: list,
              FILE_NAME="watchlist.csv") -> pd.DataFrame:

    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    df_syms = pd.read_pickle(DATAPATH.joinpath("df_symlots.pkl"))
    df_syms = df_syms[df_syms.symbol.isin(symbols)]
    df_syms["primaryExchange"] = [
        "SMART" + "/" + "AMEX" if c.exchange == "SMART" else c.exchange
        for c in df_syms.contract
    ]
    df_syms = df_syms[["symbol", "secType", "primaryExchange"]]

    # Write to file
    try:
        os.remove(DATAPATH.joinpath(FILE_NAME))
    except OSError:
        pass
    finally:
        with open(DATAPATH.joinpath(FILE_NAME), "w") as f:
            f.write("COLUMN,0\n")
            for s in df_syms.itertuples():
                f.write(f"DES,{s[1]},{s[2]},{s[3]}\n")
    return df_syms


def yes_or_no(question):
    while True:
        answer = input(question + " (y/n): ").lower().strip()
        if answer in ("y", "yes", "n", "no"):
            return answer in ("y", "yes")
        else:
            print("You must answer yes or no.")


if __name__ == "__main__":

    pass
