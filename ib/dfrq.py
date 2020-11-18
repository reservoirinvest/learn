# ** SETUP
# .Imports
import os
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
from ib_insync import IB, Contract, MarketOrder, util

from engine import Timer, Vars, executeAsync, margin, qualify, save_df
from support import quick_pf

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
VAR_YML = os.path.join(THIS_FOLDER, "var.yml")


def get_dfrq(MARKET: str) -> pd.DataFrame:
    ibp = Vars(MARKET.upper())  # IB Parameters from var.yml

    HOST, PORT, CID = ibp.HOST, ibp.PORT, ibp.CID

    LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")
    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    # * SETUP LOGS AND CLEAR THEM
    LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_dfrq.log")
    util.logToFile(path=LOGFILE, level=30)
    with open(LOGFILE, "w"):
        pass

    # ... start the time
    dfrq_time = Timer("dfrqs")
    dfrq_time.start()

    # * GET PNL AND PORTFOLIO

    # ... portfolio
    with IB().connect(ibp.HOST, ibp.PORT, ibp.CID) as ib:
        df_pf = quick_pf(ib)
        ib.disconnect()
        IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

    # * GET MARGINS CONSUMED BY PORTFOLIO
    pf_raw_cts = [Contract(conId=c) for c in df_pf.conId]

    # .qualify portfolio contracts
    with IB().connect(HOST, PORT, CID) as ib:
        pf_cts = ib.run(qualify(ib, pf_raw_cts))

    # .get orders and make cos
    df_pf = df_pf.assign(contract=pf_cts)
    df_pf = df_pf.assign(
        order=[
            MarketOrder("SELL", abs(p)) if p > 0 else MarketOrder("BUY", abs(p))
            for p in df_pf.position
        ]
    )

    cos = [(c, o) for c, o in zip(df_pf.contract, df_pf.order)]

    # .get margins
    with IB().connect(HOST, PORT, CID) as ib:
        df_pfm = ib.run(
            executeAsync(
                ib=ib,
                algo=margin,
                cts=cos,
                CONCURRENT=200,
                TIMEOUT=5,
                post_process=save_df,
            )
        )

    df_pf = (
        df_pf.set_index("conId")
        .join(df_pfm[["conId", "margin", "comm"]].set_index("conId"))
        .reset_index()
        .drop("order", 1)
    )

    # * GET GROSS POSITIONS
    # .map lots for the options
    df_symlots = pd.read_pickle(DATAPATH.joinpath("df_symlots.pkl"))
    lotmap = df_symlots[["symbol", "lot"]].set_index("symbol").to_dict("dict")["lot"]
    lot = np.where(df_pf.secType == "OPT", df_pf.symbol.map(lotmap), 1)
    df_pf.insert(7, "lot", lot)

    # .get gross position (long/short commitment)
    df_pf = df_pf.assign(
        grosspos=np.where(
            df_pf.secType == "OPT",
            df_pf.strike * df_pf.position * df_pf.lot,
            df_pf.mktPrice * df_pf.position * df_pf.lot,
        )
    )

    df_gp = df_pf.groupby("symbol").grosspos.apply(sum).sort_values(ascending=False)

    # .get lotmap from df_unds
    df_unds = pd.read_pickle(DATAPATH.joinpath("df_unds.pkl"))
    lotmap = df_symlots[["symbol", "lot"]].set_index("symbol").to_dict("dict")["lot"]
    s_gross = df_unds.close * df_unds.symbol.map(lotmap)
    dfrq = (
        df_unds.assign(lot=df_unds.symbol.map(lotmap), gross=s_gross)
        .sort_values("gross", ascending=False)[["symbol", "undPrice", "lot", "gross"]]
        .reset_index(drop=True)
    )

    # * BUILD dfrq
    # .integrate grosspos into dfrq
    dfrq = dfrq.assign(grosspos=dfrq.symbol.map(df_gp).apply(abs)).sort_values(
        "grosspos", ascending=False
    )

    MAX_GROSSPOS = max(s_gross) if MARKET == "NSE" else s_gross.quantile(0.8)

    # .compute remaining quantities from MAX_GROSSPOS
    remq = (MAX_GROSSPOS - dfrq.grosspos.fillna(0)) / (dfrq.undPrice * dfrq.lot)
    dfrq = dfrq.assign(remq=remq)

    dfrq.loc[dfrq.remq < 0, "remq"] = 0  # zerorize negative remq
    dfrq = dfrq.assign(remq=dfrq.remq.apply(round))  # round up to integer

    # .set minimum of 1 contract for high gross symbols
    dfrq.loc[(dfrq.remq == 0) & dfrq.grosspos.isnull(), "remq"] = 1

    # * DERIVE STATUSES FROM PORTFOLIO
    # * partials, fresh, undefended, uncovered, dodo (risky & uncovered), orphan, harvest and balanced

    # Partials: symbols whose long/short stock positions don't cover the lots
    m_partial = (df_pf.secType == "STK") & (
        df_pf.position % df_pf.symbol.map(lotmap) != 0
    )
    df_partial = df_pf[m_partial]
    partials = sorted(df_partial.symbol.unique())

    # Fresh: symbols with grosspos = NaN - fresh nakeds
    m_fresh = dfrq.grosspos.isnull()
    df_fresh = dfrq[m_fresh]
    fresh = sorted(df_fresh.symbol.unique())

    # Undefended, but covered: stock symbols in pf with short option positions
    m_undefended = (
        df_pf.symbol.isin(df_pf[df_pf.secType == "STK"].symbol)
        & (~df_pf.symbol.isin(partials))
        & (df_pf.secType == "OPT")
        & (df_pf.position < 0)
    )
    df_undefended = df_pf[m_undefended]
    undefended = sorted(df_undefended.symbol.unique())

    # Uncovered, but defended: stock symbols in pf with protective long option positions
    m_uncovered = (
        df_pf.symbol.isin(df_pf[df_pf.secType == "STK"].symbol)
        & (~df_pf.symbol.isin(partials))
        & (df_pf.secType == "OPT")
        & (df_pf.position > 0)
    )
    df_uncovered = df_pf[m_uncovered]
    uncovered = sorted(df_uncovered.symbol.unique())

    # dodo: non-fresh, non-partial stock symbols that are neither covered and not protected
    not_dodo = set(list(partials) + list(fresh) + list(undefended) + list(uncovered))
    m_dodo = df_pf.symbol.isin(
        df_pf[df_pf.secType == "STK"].symbol
    ) & ~df_pf.symbol.isin(not_dodo)
    df_dodo = df_pf[m_dodo]
    dodo = sorted(df_dodo.symbol.unique())

    # orphan: short calls and short longs not in `Fresh` and whithout underlying stocks
    m_orphan = (
        ~df_pf.symbol.isin(df_pf[df_pf.secType == "STK"].symbol)
        & (~df_pf.symbol.isin(partials))
        & (df_pf.secType == "OPT")
    )
    df_orphan = df_pf[m_orphan]

    # Keep only those longs without any ``compensating`` shorts.
    # ... this is done by grouping by symbols and adding the positions
    # ... sumpos > 0 are orphan which can be set up with a short near its strike.
    df_orphan = df_orphan.assign(
        sumpos=df_orphan.groupby("symbol").position.transform(sum)
    )

    df_orphan = df_orphan[df_orphan.sumpos > 0]

    orphan = sorted(df_orphan.symbol.unique())

    # harvest: the remaining symbols are the ones to be harvested
    # Note that this is reversed! First list and then the df!!
    harvest = set(df_symlots.symbol.to_list()) - set(
        partials + fresh + undefended + uncovered + dodo + orphan
    )
    df_harvest = df_pf[df_pf.symbol.isin(harvest)].sort_values("unPnL")

    # Build dictionary of statuses

    y = defaultdict(list)
    status = {
        "partials": partials,
        "fresh": fresh,
        "undefended": undefended,
        "uncovered": uncovered,
        "dodo": dodo,
        "orphan": orphan,
        "harvest": harvest,
    }
    for s in df_symlots.symbol.unique():
        for k, v in status.items():
            if s in v:
                y[s].append(k)

    # ..introduce `balanced` for symbols which are both covered and defended
    z = defaultdict()
    for k, v in y.items():
        # Note: uncovered is defended, and undefended is covered.
        # ... so checking against a set of both to get both covered and defended!!
        if set(v) == set(["undefended", "uncovered"]):
            z[k] = "balanced"
        else:
            z[k] = v[0]

    dfrq = dfrq.assign(status=dfrq.symbol.map(pd.Series(z)))
    dfrq.to_pickle(DATAPATH.joinpath("dfrq.pkl"))

    dfrq_time.stop()

    return dfrq


if __name__ == "__main__":

    MARKET = "SNP"

    dfrq = get_dfrq(MARKET)

    print(dfrq)
