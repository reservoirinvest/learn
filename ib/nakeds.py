# Generate fresh nakeds

# ** SETUP
# .Imports
import math
import os
import pathlib

import numpy as np
import pandas as pd
from ib_insync import IB, MarketOrder, util

from dfrq import get_dfrq
from engine import executeAsync, get_unds, margin, price, save_df
from support import (Timer, Vars, calcsdmult_df, get_dte, get_market, get_prec,
                     yes_or_no)


def get_nakeds(MARKET: str, RECALC_UNDS: bool = True) -> pd.DataFrame:
    ibp = Vars(MARKET.upper())  # IB Parameters from var.yml

    HOST, PORT, CID = ibp.HOST, ibp.PORT, ibp.CID
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

    LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")
    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    # * SETUP LOGS AND CLEAR THEM
    LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_nakeds.log")
    util.logToFile(path=LOGFILE, level=30)
    with open(LOGFILE, "w"):
        pass

    # . start the time
    nakeds_time = Timer("nakeds")
    nakeds_time.start()

    # * LOAD FILES
    qopts = pd.read_pickle(DATAPATH.joinpath("qopts.pkl"))
    df_symlots = pd.read_pickle(DATAPATH.joinpath("df_symlots.pkl"))
    df_chains = pd.read_pickle(DATAPATH.joinpath("df_chains.pkl"))

    # * GET df_unds AND dfrq
    und_cts = df_symlots.contract.unique()

    if RECALC_UNDS:
        df_unds = get_unds(MARKET, und_cts, savedf=True)
    else:
        df_unds = pd.read_pickle(DATAPATH.joinpath("df_unds.pkl"))

    dfrq = get_dfrq(MARKET)

    nakeds = set(dfrq[dfrq.status == "naked"].symbol)

    # . generate df_opts from qopts.pkl
    optcols = "conId,symbol,secType,lastTradeDateOrContractMonth,strike,right".split(
        ","
    )
    df_opts = util.df(qopts.to_list())[optcols].rename(
        columns={"lastTradeDateOrContractMonth": "expiry"}
    )

    qo_dict = {int(q.conId): q for q in qopts}
    df_opts["contract"] = df_opts.conId.map(qo_dict)

    df_opts = df_opts.dropna(subset=["contract"]).reset_index(
        drop=True
    )  # Remove NaN in contracts!

    # * BUILD df_nakeds AND SCRUB IT UP

    # . build df_nakeds
    df_nakeds = df_opts[df_opts.symbol.isin(nakeds)]

    df_nakeds = df_nakeds[~df_nakeds.symbol.isin(
        ibp.BLACKLIST)]  # remove blacklist

    # . remove dtes
    df_nakeds.insert(3, "dte", df_nakeds.expiry.apply(get_dte))
    df_nakeds = df_nakeds[df_nakeds.dte.between(
        ibp.MINDTE, ibp.MAXDTE, inclusive=True)]

    # .remove options within stdev fence

    # . integrate undPrice and und_iv
    df_nakeds = (
        df_nakeds.set_index("symbol")
        .join(df_unds.set_index("symbol")[["iv", "undPrice"]])
        .reset_index()
    )

    # . compute One stdev
    df_nakeds = df_nakeds.assign(
        sd1=df_nakeds.undPrice * df_nakeds.iv *
        (df_nakeds.dte / 365).apply(math.sqrt)
    )

    hi_sd = df_nakeds.undPrice + df_nakeds.sd1 * ibp.CALLSTDMULT
    lo_sd = df_nakeds.undPrice - df_nakeds.sd1 * ibp.PUTSTDMULT

    df_nakeds = df_nakeds.assign(hi_sd=hi_sd, lo_sd=lo_sd)

    # . remove options within stdev fence
    fence_mask = ((df_nakeds.strike > hi_sd) & (df_nakeds.right == "C")) | (
        (df_nakeds.strike < lo_sd) & (df_nakeds.right == "P")
    )

    df_nakeds = df_nakeds[fence_mask]

    # .remove options outside of remqty / MAXOPTQTY_SYM
    # .... map symbol to remqty
    remq = dfrq.set_index("symbol").remq.to_dict()

    # ... limit remqty to MAXOPTQTY_SYM
    remq = {k: min(v, ibp.MAXOPTQTY_SYM) for k, v in remq.items()}

    # ... sort and pick top options around the fence [ref: https://stackoverflow.com/questions/64864630]
    # . reverse strike for Calls to get the right sort order for top values
    df = df_nakeds.assign(
        value=np.where(df_nakeds.right == "C", -1 *
                       df_nakeds.strike, df_nakeds.strike)
    )

    # . build cumcount series with index aligned to df
    s = (
        (df.sort_values(["symbol", "dte", "value"]).groupby(
            ["symbol", "dte", "right"]))
        .cumcount()
        .reindex(df.index)
    )

    # . get remq for symbols from df/series and pick up the top cumcounts
    df_nakeds1 = (
        df[s < df.symbol.map(remq)]
        .sort_values(["symbol", "dte", "value"], ascending=[True, True, False])
        .drop("value", 1)
        .reset_index(drop=True)
    )

    # . remove sd1 and rename `iv` to `und_iv`
    df_nakeds1 = df_nakeds1.drop("sd1", 1).rename(columns={"iv": "und_iv"})

    # . integrate lots
    idx_cols = ["symbol", "expiry", "strike"]
    df_nakeds1 = (
        df_nakeds1.set_index(idx_cols)
        .join(df_chains.set_index(idx_cols).lot)
        .reset_index()
    )

    # * GET PRICE, IV AND MARGIN OF OPTIONS AND INTEGRATE

    opt_price_time = Timer(f"{MARKET} nakeds option price")
    opt_price_time.start()

    nakeds_contracts = df_nakeds1.contract.to_list()

    nakeds_orders = [
        MarketOrder("SELL", lot / lot)
        if MARKET.upper() == "SNP"
        else MarketOrder("SELL", lot)
        for lot in df_nakeds1.lot
    ]

    # . get nakeds option price and iv (with warning Best: 44*4, TIMEOUT=11, `FILL_DELAY`: 11)
    with IB().connect(HOST, PORT, CID) as ib:
        df_opt_prices = ib.run(
            executeAsync(
                ib=ib,
                algo=price,
                cts=nakeds_contracts,
                CONCURRENT=40,
                TIMEOUT=8,
                post_process=save_df,
                DATAPATH=DATAPATH,
                **{"FILL_DELAY": 5.5},
                OP_FILENAME="",
                SHOW_TQDM=True,
            )
        )

        # to prevent first TimeoutError()
        ib.disconnect()
        IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

    opt_price_time.stop()

    # . get nakeds option margins (Best: 50*4, TIMEOUT=5, `FILL_DELAY`: 5)

    opt_margin_time = Timer(f"{MARKET} naked option margins")
    opt_margin_time.start()

    opt_cos = [(c, o) for c, o in zip(nakeds_contracts, nakeds_orders)]

    with IB().connect(HOST, PORT, CID) as ib:
        df_opt_margins = ib.run(
            executeAsync(
                ib=ib,
                algo=margin,
                cts=opt_cos,
                CONCURRENT=200,
                TIMEOUT=5.5,
                post_process=save_df,
                DATAPATH=DATAPATH,
                OP_FILENAME="",
                SHOW_TQDM=True,
                **{"FILL_DELAY": 5.5},
            )
        )

    opt_margin_time.stop()

    # * GET ROM AND SET EXPECTED OPTION PRICE

    # . integrate price, iv and margins
    df_nakeds2 = (
        df_nakeds1.set_index("conId")
        .join(df_opt_margins.set_index("conId")[["comm", "margin"]])
        .join(
            df_opt_prices.set_index("conId")[
                ["bid", "ask", "close", "last", "iv", "price"]
            ]
        )
        .reset_index()
    )

    # . update null iv with und_iv
    m_iv = df_nakeds2.iv.isnull()
    df_nakeds2.loc[m_iv, "iv"] = df_nakeds2[m_iv].und_iv

    # . update calculated sd mult for strike
    df_nakeds2.insert(19, "sdMult", calcsdmult_df(
        df_nakeds2.strike, df_nakeds2))

    # . put the order quantity
    df_nakeds2["qty"] = 1 if MARKET == "SNP" else df_nakeds2.lot

    # . fill empty commissions
    if MARKET == "NSE":
        commission = 20.0
    else:
        commission = 2.0

    df_nakeds2["comm"].fillna(value=commission, inplace=True)

    # ... add intrinsic and time values
    df_nakeds2 = df_nakeds2.assign(
        intrinsic=np.where(
            df_nakeds2.right == "C",
            (df_nakeds2.undPrice - df_nakeds2.strike).clip(0, None),
            (df_nakeds2.strike - df_nakeds2.undPrice).clip(0, None),
        )
    )
    df_nakeds2 = df_nakeds2.assign(
        timevalue=df_nakeds2.price - df_nakeds2.intrinsic)

    # . compute rom based on timevalue, remove zero rom and down-sort on it
    df_nakeds2["rom"] = (
        (df_nakeds2.timevalue * df_nakeds2.lot - df_nakeds2.comm).clip(0)
        / df_nakeds2.margin
        * 365
        / df_nakeds2.dte
    )

    df_nakeds2 = (
        df_nakeds2[df_nakeds2.rom > 0]
        .sort_values("rom", ascending=False)
        .reset_index(drop=True)
    )

    # .establish expRom
    #    ... for those whose RoM is < MINROM, make it equal to MINROM
    df_nakeds2["expRom"] = np.maximum(ibp.MINEXPROM, df_nakeds2.rom)

    # . set expPrice to be based on expRom
    df_nakeds2["expPrice"] = (
        df_nakeds2.expRom
        * np.maximum(ibp.MINOPTSELLPRICE, df_nakeds2.price)
        / df_nakeds2.rom
    ).apply(lambda x: get_prec(x, ibp.PREC))

    # . remove NaN from expPrice
    df_nakeds2 = df_nakeds2.dropna(subset=["expPrice"]).reset_index(drop=True)

    # * PICKLE AND SAVE TO EXCEL
    df_nakeds2.to_pickle(DATAPATH.joinpath("df_nakeds.pkl"))

    df_calls = df_nakeds2[df_nakeds2.right == "C"].reset_index(drop=True)
    df_puts = df_nakeds2[df_nakeds2.right == "P"].reset_index(drop=True)

    # ... initiate Excel writer object
    writer = pd.ExcelWriter(DATAPATH.joinpath(
        "df_nakeds.xlsx"), engine="xlsxwriter")

    df_nakeds2.to_excel(
        writer, sheet_name="All", float_format="%.2f", index=False, freeze_panes=(1, 1)
    )

    df_calls.to_excel(
        writer,
        sheet_name="Calls",
        float_format="%.2f",
        index=False,
        freeze_panes=(1, 1),
    )

    df_puts.to_excel(
        writer, sheet_name="Puts", float_format="%.2f", index=False, freeze_panes=(1, 1)
    )

    all_sheet = writer.sheets["All"]
    puts_sheet = writer.sheets["Calls"]
    calls_sheet = writer.sheets["Puts"]
    sheets = [all_sheet, puts_sheet, calls_sheet]

    for sht in sheets:
        # Hide all rows without data
        sht.set_default_row(hide_unused_rows=True)

        sht.set_column("A:A", None, None, {"hidden": True})  # Hide conId
        sht.set_column("H:H", None, None, {"hidden": True})  # Hide contract

    try:
        writer.save()
    except Exception as e:
        print(f"\nError {e}: df_nakeds.xlsx is open or has some issues!!!\n")

    nakeds_time.stop()

    return df_nakeds2


if __name__ == "__main__":

    MARKET = get_market("Create nakeds options for:")
    RECALC_UNDS = yes_or_no('Want to recaculate underlyings? ')

    get_nakeds(MARKET=MARKET, RECALC_UNDS=RECALC_UNDS)
