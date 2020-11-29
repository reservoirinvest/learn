# ** SETUP
# .Imports
import os
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
from ib_insync import IB, Contract, MarketOrder, util

from engine import Timer, Vars, executeAsync, margin, qualify, save_df
from support import get_market, quick_pf

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

    # ... load files
    df_unds = pd.read_pickle(DATAPATH.joinpath("df_unds.pkl"))
    df_symlots = pd.read_pickle(DATAPATH.joinpath("df_symlots.pkl"))

    # ... portfolio
    with IB().connect(ibp.HOST, ibp.PORT, ibp.CID) as ib:
        df_pf = quick_pf(ib)
        ib.disconnect()
        IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

    # * GET MARGINS CONSUMED BY PORTFOLIO

    if not df_pf.empty:
        pf_raw_cts = [Contract(conId=c) for c in df_pf.conId]

        # .qualify portfolio contracts
        with IB().connect(HOST, PORT, CID) as ib:
            pf_cts = ib.run(qualify(ib, pf_raw_cts))

        # .get orders and make cos
        df_pf = df_pf.assign(contract=pf_cts)
        df_pf = df_pf.assign(
            order=[
                MarketOrder("SELL", abs(p)) if p > 0 else MarketOrder(
                    "BUY", abs(p))
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
                    OP_FILENAME="",
                    **{"FILL_DELAY": 5},
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
        lotmap = df_symlots[["symbol", "lot"]].set_index(
            "symbol").to_dict("dict")["lot"]
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

    else:

        # ...extend the empty df_pf
        df_pf = df_pf.assign(margin=np.nan, comm=np.nan, grosspos=np.nan)

    df_gp = df_pf.groupby("symbol").grosspos\
        .apply(sum).sort_values(ascending=False)

    # .get lotmap from df_unds

    lotmap = df_symlots[["symbol", "lot"]].set_index(
        "symbol").to_dict("dict")["lot"]
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
    remq = (MAX_GROSSPOS - dfrq.grosspos.fillna(0)) / \
        (dfrq.undPrice * dfrq.lot)
    dfrq = dfrq.assign(remq=remq)

    dfrq.loc[dfrq.remq < 0, "remq"] = 0  # zerorize negative remq
    dfrq = dfrq.assign(remq=dfrq.remq.apply(round))  # round up to integer

    # .set minimum of 1 contract for high gross symbols
    dfrq.loc[(dfrq.remq == 0) & dfrq.grosspos.isnull(), "remq"] = 1

    # * DERIVE STATUSES FROM PORTFOLIO
    # * partials, naked, undefended, uncovered, dodo (risky & uncovered), orphan, harvest and balanced

    # ... get the stocks and options
    df_stk = df_pf[df_pf.secType.isin(["STK", "IND"])].reset_index(drop=True)
    df_opt = df_pf[df_pf.secType == "OPT"].reset_index(drop=True)

    # Partials: symbols whose long/short stock positions don't cover the lots
    m_partial = (df_pf.secType.isin(["STK", "IND"])) & (
        df_pf.position % df_pf.symbol.map(lotmap) != 0
    )
    df_partial = df_pf[m_partial]
    partials = sorted(df_partial.symbol.unique())

    # Nakeds: symbols with grosspos = NaN - fresh nakeds
    m_naked = dfrq.grosspos.isnull()
    df_naked = dfrq[m_naked]
    naked = sorted(df_naked.symbol.unique())

    # Orphan: long calls and long puts not in `naked` and whithout underlying stocks
    m_orphan = (
        (df_opt.position > 0)
        & ~df_opt.symbol.isin(naked)
        & ~df_opt.symbol.isin(df_stk.symbol.unique())
    )

    df_orphan = df_opt[m_orphan]

    orphan = sorted(df_orphan.symbol.unique())

    # Uncovered: stock symbols in pf without short covered call / put positions
    m_covered = (
        (df_opt.position < 0)
        & df_opt.symbol.isin(df_stk.symbol)
        & ~df_opt.symbol.isin(partials)
    )
    already_covered = df_opt[m_covered].symbol.to_list()

    df_uncovered = df_stk[~df_stk.symbol.isin(already_covered)]
    uncovered = sorted(df_uncovered.symbol.unique())

    # Undefended: stock symbols in pf without long protective call / put options
    m_defended = (
        (df_opt.position > 0)
        & df_opt.symbol.isin(df_stk.symbol)
        & ~df_opt.symbol.isin(partials)
    )
    already_defended = df_opt[m_defended].symbol.to_list()

    df_undefended = df_stk[~df_stk.symbol.isin(already_defended)]
    undefended = sorted(df_undefended.symbol.unique())

    # dodo: stock symbols that are neither covered and not protected
    dodo = [s for s in uncovered if s in undefended]

    # balanced: stock symbols that are both covered and protected
    balanced = [s for s in already_covered if s in already_defended]

    # remove dodos and balanced from uncovered and undefended
    uncovered = [s for s in uncovered if s not in set(dodo + balanced)]
    undefended = [s for s in undefended if s not in set(dodo + balanced)]

    # harvest: symbols not in all other statuses
    harvest = set(dfrq.symbol) - set(
        orphan + uncovered + undefended + dodo + partials + naked + balanced
    )

    # map the status to dfrq symbols

    status_dict = {
        "naked": naked,
        "orphan": orphan,
        "uncovered": uncovered,
        "undefended": undefended,
        "dodo": dodo,
        "balanced": balanced,
        "partials": partials,
        "harvest": harvest,
    }

    status = dict()
    for k, v in status_dict.items():
        for i in v:
            status[i] = k

    dfrq["status"] = dfrq.symbol.map(status)

    dfrq.to_pickle(DATAPATH.joinpath("dfrq.pkl"))

    dfrq_time.stop()

    return dfrq


if __name__ == "__main__":

    MARKET = get_market("Make dfrqs for:")

    dfrq = get_dfrq(MARKET)

    print(dfrq)
