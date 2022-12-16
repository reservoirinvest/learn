# proposes defensive trades
import math
import os
import pathlib
import pickle

import numpy as np
import pandas as pd
from ib_insync import IB, MarketOrder, Option, util

from dfrq import get_dfrq
from engine import get_chains, get_margins, get_prices, get_unds, qualify
from support import Timer, Vars, calcsdmult_df, get_prec, quick_pf


def get_defends(MARKET: str = "SNP", RUN_ON_PAPER=False, SAVEXL: bool = True):

    defends_time = Timer(f"{MARKET} defends time")
    defends_time.start()

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
    LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_defends.log")
    util.logToFile(path=LOGFILE, level=30)
    with open(LOGFILE, "w"):
        pass

    # ... file loads
    df_symlots = pd.read_pickle(DATAPATH.joinpath("df_symlots.pkl"))

    # * GET THE UNDEFENDED STOCKS FROM DFRQ
    dfrq = get_dfrq(MARKET)

    # ... get the portfolio, chains and unds for uncovered stocks
    with IB().connect(HOST, PORT, CID) as ib:
        df_pf = quick_pf(ib)

    undef = dfrq[dfrq.status.isin(["dodo", "undefended"])].symbol.unique()
    undef_cts = set(df_symlots[df_symlots.symbol.isin(undef)].contract.unique())

    df_ch = get_chains(MARKET=MARKET, und_cts=undef_cts, SAVE=False, RUN_ON_PAPER=False)
    df_u = get_unds(MARKET=MARKET, und_cts=undef_cts, SAVE=False, RUN_ON_PAPER=False)

    # * BUILD CHAINS

    # ... filter chains more than 6 months away
    df_ch1 = df_ch[df_ch.dte > eval(ibp.DEFEND_DTE)]
    df_ch2 = df_ch1[df_ch1.dte == df_ch1.groupby("symbol").dte.transform(min)]

    # ... integrate iv
    df_ch2 = (
        df_ch2.set_index("symbol")
        .join(df_u[["symbol", "undPrice", "iv"]].set_index("symbol"))
        .rename(columns={"iv": "und_iv"})
    )

    # ... integrate portfolio
    m = (df_pf.secType == "STK") & (df_pf.symbol.isin(undef))
    df_ch3 = df_ch2.join(
        df_pf[m][["symbol", "position", "avgCost"]].set_index("symbol")
    ).reset_index()

    # For positions < 0, we need defensive calls. For positions > 0 we need defensive puts.
    df_defends = df_ch3.assign(right=np.where(df_ch3.position > 0, "P", "C"))

    # Filter out defend options whose strikes are beyond the defense threshold
    # ... make strike delta
    df_defends = df_defends.assign(
        strkDelta=df_defends.undPrice
        * df_defends.und_iv
        * (df_defends.dte / 365).apply(math.sqrt)
    )

    # ... get the strike reference based the threshold

    strikeRef = np.where(
        df_defends.right == "C",
        df_defends.undPrice + ibp.DEFEND_TH * df_defends.strkDelta,
        df_defends.undPrice - ibp.DEFEND_TH * df_defends.strkDelta,
    )

    df_defends = df_defends.assign(strikeRef=strikeRef).reset_index(drop=True)

    # ... get strikes closest to strikeRef
    df_defends = (
        df_defends.groupby("symbol", as_index=False)
        .apply(lambda g: g.iloc[abs(g.strikeRef - g.strike).argsort()[:12]])
        .drop(["strkDelta", "strikeRef"], 1)
        .reset_index(drop=True)
    )

    # ... set the action as BUY for defends
    df_defends = df_defends.assign(action="BUY")

    # ... make the quantity
    df_defends = df_defends.assign(
        qty=(df_defends.position / df_defends.lot).apply(abs)
    )

    # * GET THE CONTRACTS

    # ... build the contracts
    optcon_raw = [
        Option(s, e, k, r, "SMART")
        for s, e, k, r in zip(
            df_defends.symbol, df_defends.expiry, df_defends.strike, df_defends.right
        )
    ]

    with IB().connect(HOST, PORT, CID) as ib:
        dfoc = ib.run(qualify(ib, optcon_raw))

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
        df_defends.set_index(cols)
        .join(dfoc[cols + ["conId", "contract"]].set_index(cols))
        .dropna()
        .reset_index()
    )

    # ... get the prices
    dfop = get_prices(MARKET=MARKET, cts=df_ch6.contract, FILL_DELAY=8)

    # ... get margins
    orders = [
        MarketOrder(action, qty) for action, qty in zip(df_ch6.action, df_ch6.qty)
    ]

    cos = list(zip(df_ch6.contract, orders))

    dfom = get_margins(MARKET, cos, msg="defend margins")

    # ... integrate price and margins
    df_ch7 = (
        df_ch6.set_index("conId")
        .join(dfop[["conId", "bid", "ask", "close", "price", "iv"]].set_index("conId"))
        .join(dfom[["conId", "margin", "comm"]].set_index("conId"))
        .reset_index()
    )

    # * SELECT THE DEFENSES

    # ... replace iv with und_iv where it is not available
    df_ch8 = df_ch7.assign(iv=df_ch7.iv.fillna(df_ch7.und_iv))

    # ... replace iv with und_iv where it is not available
    df_ch8 = df_ch7.assign(iv=df_ch7.iv.fillna(df_ch7.und_iv))

    # .. set the expected price
    df_ch8["expPrice"] = (df_ch8.price - 3 * ibp.PREC).apply(
        lambda x: get_prec(x, ibp.PREC)
    )
    df_ch8 = df_ch8.assign(sdmult=calcsdmult_df(df_ch8.strike, df_ch8))

    # ... make the action as 'BUY'
    df_ch8["action"] = "BUY"

    # ... stage the columns
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
        "sdmult",
        "lot",
        "margin",
        "action",
        "qty",
        "price",
        "expPrice",
    ]
    df_ch8 = df_ch8[cols]

    df_ch8["def_pct"] = np.where(
        df_ch8.right == "C",
        (df_ch8.strike - df_ch8.undPrice) / df_ch8.undPrice,
        (df_ch8.undPrice - df_ch8.strike) / df_ch8.undPrice,
    )

    df_ch8["defense"] = df_ch8.def_pct * df_ch8.undPrice * df_ch8.lot * df_ch8.qty

    df_ch8 = df_ch8.assign(cost=df_ch8.expPrice * df_ch8.lot * df_ch8.qty)

    df_raw_defends = df_ch8  # The entire set for alternatives

    # * CHOOSE DEFENSE WITH THE LOWEST COST
    df_defends = df_ch8[df_ch8.cost == df_ch8.groupby("symbol").cost.transform(min)]

    # ... report missing underlying symbols
    missing_und_symbols = [
        s for s in undef if s not in list(df_defends.symbol.unique())
    ]

    if missing_und_symbols:
        print(f"\nNote: {missing_und_symbols} options could not be defended...\n")

    # ... pickle and save to Excel
    saveObject = {
        "df_defends": df_defends,
        "df_raw_defends": df_raw_defends,
        "missing": missing_und_symbols,
    }

    with open(DATAPATH.joinpath("df_defends.pkl"), "wb") as f:
        pickle.dump(saveObject, f)

    if SAVEXL:
        writer = pd.ExcelWriter(DATAPATH.joinpath("propose_defends.xlsx"))
        df_defends.to_excel(
            writer,
            sheet_name="Defends",
            float_format="%.2f",
            index=False,
            freeze_panes=(1, 1),
        )
        df_raw_defends.to_excel(
            writer,
            sheet_name="Alternatives",
            float_format="%.2f",
            index=False,
            freeze_panes=(1, 1),
        )
        sht1 = writer.sheets["Defends"]
        sht2 = writer.sheets["Alternatives"]
        sht1.set_column("A:B", None, None, {"hidden": True})
        sht2.set_column("A:B", None, None, {"hidden": True})
        writer.save()

    defends_time.stop()

    return df_defends


if __name__ == "__main__":
    get_defends()
