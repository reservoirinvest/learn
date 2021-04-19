# proposes cover trades

import math
import os
import pathlib
import pickle

import numpy as np
import pandas as pd
from ib_insync import IB, MarketOrder, Option, util

from dfrq import get_dfrq
from engine import get_chains, get_margins, get_prices, get_unds, qualify
from support import Timer, Vars, calcsdmult_df, get_prec, get_prob, quick_pf

import requests


# Get covers for stocks
def get_covers(
    MARKET: str = "SNP",  # currently valid only for SNP!
    RUN_ON_PAPER: bool = False,
    COVERSD: float = 1.3,
    COV_DEPTH: int = 4,
    SAVEXL: bool = True,
) -> pd.DataFrame:

    covers_time = Timer(f"{MARKET} cover time")
    covers_time.start()

    # * SETUP
    ibp = Vars(MARKET.upper())  # IB Parameters from var.yml
    HOST, CID = ibp.HOST, ibp.CID
    if RUN_ON_PAPER:
        PORT = ibp.PAPER
    else:
        PORT = ibp.PORT

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")
    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    # ...setup logs and clear them
    LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_covers.log")
    util.logToFile(path=LOGFILE, level=30)
    with open(LOGFILE, "w"):
        pass

    # ... file loads
    df_symlots = pd.read_pickle(DATAPATH.joinpath("df_symlots.pkl"))

    # * GET STOCKS NOT COVERED FROM DFRQ
    dfrq = get_dfrq(MARKET)

    IB().disconnect()
    IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

    # ... get the portfolio, chains and unds for uncovered stocks
    with IB().connect(HOST, PORT, CID) as ib:
        df_pf = quick_pf(ib)

    IB().disconnect()
    IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

    uncov = dfrq[(dfrq.status == "uncovered") | (dfrq.status == "dodo")].symbol.unique()
    uncov_cts = df_symlots[df_symlots.symbol.isin(uncov)].contract.to_list()

    df_ch = get_chains(MARKET=MARKET, und_cts=uncov_cts, SAVE=False, RUN_ON_PAPER=False)
    IB().disconnect()
    IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

    df_u = get_unds(MARKET=MARKET, und_cts=uncov_cts, SAVE=False, RUN_ON_PAPER=False)
    IB().disconnect()
    IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

    # * BUILD THE OPTION CONTRACTS

    # ... get the next week (after 5 days) chains
    m = df_ch.dte > 5
    min_dte = df_ch[m].groupby("symbol").dte.transform(min)

    df_ch1 = df_ch[m][df_ch[m].dte == min_dte].drop("exchange", 1)

    # ... integrate iv and portfolio
    df_ch2 = (
        df_ch1.set_index("symbol")
        .join(df_u[["symbol", "undPrice", "iv"]].set_index("symbol"))
        .rename(columns={"iv": "und_iv"})
    )

    # ... integrate portfolio
    m = (df_pf.secType == "STK") & (df_pf.symbol.isin(uncov))
    df_ch3 = df_ch2.join(
        df_pf[m][["symbol", "position", "avgCost"]].set_index("symbol")
    ).reset_index()

    # ... set the right based on position for the cover
    df_ch3["right"] = np.where(df_ch3.position > 0, "C", "P")

    # * GET OPTION PRICES AND MARGINS

    # Get options for the closest strike
    # ... get the strkDelta for strikeRef
    # ... note that it is positive for Calls and negative for Puts

    strkDelta = np.where(
        df_ch3.right == "C",
        df_ch3.undPrice * df_ch3.und_iv * (df_ch3.dte / 365).apply(math.sqrt),
        -df_ch3.undPrice * df_ch3.und_iv * (df_ch3.dte / 365).apply(math.sqrt),
    )

    df_ch3 = df_ch3.assign(strkDelta=strkDelta)

    # ... make the strikeRef for calls
    df_ch3 = df_ch3.assign(strikeRef=df_ch3.undPrice + COVERSD * df_ch3.strkDelta)

    # ... get COV_DEPTH strikes closest to the strikeRef
    df_ch5 = (
        df_ch3.groupby("symbol", as_index=False)
        .apply(lambda g: g.iloc[abs(g.strike - g.strikeRef).argsort()[:COV_DEPTH]])
        .reset_index(drop=True)
    )

    # ... determine the option quantities needed
    df_ch5["qty"] = (df_ch5.position / df_ch5.lot).apply(abs).astype("int")

    # ... build the contracts
    optcon_raw = [
        Option(s, e, k, r, "SMART")
        for s, e, k, r in zip(df_ch5.symbol, df_ch5.expiry, df_ch5.strike, df_ch5.right)
    ]

    with IB().connect(HOST, PORT, CID) as ib:
        dfoc = ib.run(qualify(ib, optcon_raw))

    IB().disconnect()
    IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

    dfoc = (
        util.df(dfoc.to_list())
        .iloc[:, :6]
        .rename(columns={"lastTradeDateOrContractMonth": "expiry"})
        .assign(contract=dfoc)
        .dropna()
        .reset_index(drop=True)
    )

    # ... integrate with the chains
    cols = ["symbol", "strike", "expiry", "right"]
    df_ch6 = (
        df_ch5.set_index(cols)
        .join(dfoc[cols + ["conId", "contract"]].set_index(cols))
        .dropna()
        .reset_index()
    )

    # ... get the prices
    dfop = get_prices(MARKET=MARKET, cts=df_ch6.contract, FILL_DELAY=8)

    # ... get margins
    orders = [MarketOrder("SELL", qty) for qty in df_ch6.qty]

    cos = list(zip(df_ch6.contract, orders))

    dfom = get_margins(MARKET, cos, msg="cover margins")

    # ... integrate price and margins
    df_ch7 = (
        df_ch6.set_index("conId")
        .join(dfop[["conId", "bid", "ask", "close", "price", "iv"]].set_index("conId"))
        .join(dfom[["conId", "margin", "comm"]].set_index("conId"))
        .reset_index()
    )

    # ... replace iv with und_iv where it is not available
    df_ch8 = df_ch7.assign(iv=df_ch7.iv.fillna(df_ch7.und_iv))

    # Set the expected price
    df_ch8["expPrice"] = (df_ch8.price + 3 * ibp.PREC).apply(
        lambda x: get_prec(x, ibp.PREC)
    )
    df_ch8 = df_ch8.assign(sdmult=calcsdmult_df(df_ch8.strike, df_ch8))
    df_ch8 = df_ch8.assign(prop=df_ch8.sdmult.apply(get_prob))

    # change conId float -> int
    df_ch8 = df_ch8.assign(conId=df_ch8.conId.astype("int"))

    # Make the action as SELL
    df_ch8["action"] = "SELL"

    # ... set cover price to MINOPTSELLPRICE
    df_ch8.loc[df_ch8.expPrice < ibp.MINOPTSELLPRICE, "expPrice"] = ibp.MINOPTSELLPRICE

    # Stage the cover columns
    cols = [
        "conId",
        "contract",
        "symbol",
        "right",
        "strike",
        "avgCost",
        "undPrice",
        "expiry",
        "dte",
        "iv",
        "strikeRef",
        "sdmult",
        "prop",
        "lot",
        "margin",
        "action",
        "qty",
        "price",
        "expPrice",
    ]
    df_ch8 = df_ch8[cols]

    df_ch8 = df_ch8.assign(revenue=df_ch8.price * df_ch8.lot * df_ch8.qty)

    df_ch8 = df_ch8.assign(
        pnl=np.where(
            df_ch8.right == "C",
            (df_ch8.strike - df_ch8.avgCost) * df_ch8.lot + df_ch8.revenue,
            (df_ch8.avgCost - df_ch8.strike) * df_ch8.lot + df_ch8.revenue,
        )
    )

    df_raw_covers = df_ch8  # The entire set

    # Choose the contracts giving the greatest profit
    df_covers = df_raw_covers[
        df_raw_covers.pnl == df_raw_covers.groupby("symbol").pnl.transform(max)
    ].reset_index(drop=True)

    # ... report missing underlying symbols
    missing_und_symbols = [s for s in uncov if s not in list(df_covers.symbol.unique())]

    if missing_und_symbols:
        print(
            f"\nNote: {missing_und_symbols} options could not be qualified for cover...\n"
        )

    # ...pickle
    saveObject = {
        "df_covers": df_covers,
        "df_raw_covers": df_raw_covers,
        "missing": missing_und_symbols,
    }

    with open(DATAPATH.joinpath("df_covers.pkl"), "wb") as f:
        pickle.dump(saveObject, f)

    if SAVEXL:
        writer = pd.ExcelWriter(DATAPATH.joinpath("propose_covers.xlsx"))
        df_covers.to_excel(
            writer,
            sheet_name="Covers",
            float_format="%.2f",
            index=False,
            freeze_panes=(1, 1),
        )
        df_raw_covers.to_excel(
            writer,
            sheet_name="Alternatives",
            float_format="%.2f",
            index=False,
            freeze_panes=(1, 1),
        )
        sht1 = writer.sheets["Covers"]
        sht2 = writer.sheets["Alternatives"]
        # sht1.set_column("A:B", None, None, {"hidden": True})
        # sht2.set_column("A:B", None, None, {"hidden": True})
        writer.save()

    covers_time.stop()

    return df_covers


if __name__ == "__main__":
    y = get_covers(COVERSD=1.4, SAVEXL=True, COV_DEPTH=4)
    print(f"\nExpected pnl is: {y.pnl.sum()}\n")
