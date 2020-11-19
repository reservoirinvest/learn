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
from support import Timer, Vars, calcsdmult_df, get_dte, get_prec


def get_fresh(MARKET: str, RECALC_UNDS: bool = True) -> pd.DataFrame:
    ibp = Vars(MARKET.upper())  # IB Parameters from var.yml

    HOST, PORT, CID = ibp.HOST, ibp.PORT, ibp.CID
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

    LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")
    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    # * SETUP LOGS AND CLEAR THEM
    LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_fresh.log")
    util.logToFile(path=LOGFILE, level=30)
    with open(LOGFILE, "w"):
        pass

    # . start the time
    fresh_time = Timer("Fresh")
    fresh_time.start()

    # * LOAD FILES
    qopts = pd.read_pickle(DATAPATH.joinpath("qopts.pkl"))
    df_symlots = pd.read_pickle(DATAPATH.joinpath("df_symlots.pkl"))
    df_chains = pd.read_pickle(DATAPATH.joinpath('df_chains.pkl'))

    # * GET df_unds AND dfrq
    und_cts = df_symlots.contract.unique()

    if RECALC_UNDS:
        df_unds = get_unds(MARKET, und_cts, savedf=True)
    else:
        df_unds = pd.read_pickle(DATAPATH.joinpath('df_unds.pkl'))

    dfrq = get_dfrq(MARKET)

    fresh = set(dfrq[dfrq.status == "fresh"].symbol)

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

    # * BUILD df_fresh AND SCRUB IT UP

    # . build df_fresh
    df_fresh = df_opts[df_opts.symbol.isin(fresh)]

    df_fresh = df_fresh[~df_fresh.symbol.isin(
        ibp.BLACKLIST)]  # remove blacklist

    # . remove dtes
    df_fresh.insert(3, "dte", df_fresh.expiry.apply(get_dte))
    df_fresh = df_fresh[df_fresh.dte.between(
        ibp.MINDTE, ibp.MAXDTE, inclusive=True)]

    # .remove options within stdev fence

    # . integrate undPrice and und_iv
    df_fresh = (
        df_fresh.set_index("symbol")
        .join(df_unds.set_index("symbol")[["iv", "undPrice"]])
        .reset_index()
    )

    # . compute One stdev
    df_fresh = df_fresh.assign(
        sd1=df_fresh.undPrice * df_fresh.iv *
        (df_fresh.dte / 365).apply(math.sqrt)
    )

    hi_sd = df_fresh.undPrice + df_fresh.sd1 * ibp.CALLSTDMULT
    lo_sd = df_fresh.undPrice - df_fresh.sd1 * ibp.PUTSTDMULT

    df_fresh = df_fresh.assign(hi_sd=hi_sd, lo_sd=lo_sd)

    # . remove options within stdev fence
    fence_mask = ((df_fresh.strike > hi_sd) & (df_fresh.right == "C")) | (
        (df_fresh.strike < lo_sd) & (df_fresh.right == "P")
    )

    df_fresh = df_fresh[fence_mask]

    # .remove options outside of remqty / MAXOPTQTY_SYM
    # .... map symbol to remqty
    remq = dfrq.set_index("symbol").remq.to_dict()

    # ... limit remqty to MAXOPTQTY_SYM
    remq = {k: min(v, ibp.MAXOPTQTY_SYM) for k, v in remq.items()}

    # ... sort and pick top options around the fence [ref: https://stackoverflow.com/questions/64864630]
    # . reverse strike for Calls to get the right sort order for top values
    df = df_fresh.assign(
        value=np.where(df_fresh.right == "C", -1 *
                       df_fresh.strike, df_fresh.strike)
    )

    # . build cumcount series with index aligned to df
    s = (
        (df.sort_values(["symbol", "dte", "value"]).groupby(
            ["symbol", "dte", "right"]))
        .cumcount()
        .reindex(df.index)
    )

    # . get remq for symbols from df/series and pick up the top cumcounts
    df_fresh1 = (
        df[s < df.symbol.map(remq)]
        .sort_values(["symbol", "dte", "value"], ascending=[True, True, False])
        .drop("value", 1)
        .reset_index(drop=True)
    )

    # . remove sd1 and rename `iv` to `und_iv`
    df_fresh1 = df_fresh1.drop("sd1", 1).rename(columns={"iv": "und_iv"})

    # . integrate lots
    idx_cols = ["symbol", "expiry", "strike"]
    df_fresh1 = (
        df_fresh1.set_index(idx_cols)
        .join(df_chains.set_index(idx_cols).lot)
        .reset_index()
    )

    # * GET PRICE, IV AND MARGIN OF OPTIONS AND INTEGRATE

    opt_price_time = Timer(f"{MARKET} fresh option price")
    opt_price_time.start()

    fresh_contracts = df_fresh1.contract.to_list()
    fresh_orders = [
        MarketOrder("SELL", lot / lot)
        if MARKET.upper() == "SNP"
        else MarketOrder("SELL", lot)
        for lot in df_fresh1.lot
    ]

    # . get fresh option price and iv (with warning Best: 44*4, TIMEOUT=11, `FILL_DELAY`: 11)
    with IB().connect(HOST, PORT, CID) as ib:
        df_opt_prices = ib.run(
            executeAsync(
                ib=ib,
                algo=price,
                cts=fresh_contracts,
                CONCURRENT=40 * 4,
                TIMEOUT=11,
                post_process=save_df,
                DATAPATH=DATAPATH,
                **{'FILL_DELAY': 11},
                OP_FILENAME="",
            )
        )

        # to prevent first TimeoutError()
        ib.disconnect()
        IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

    opt_price_time.stop()

    # . get fresh option margins (Best: 50*4, TIMEOUT=5, `FILL_DELAY`: 5)

    opt_margin_time = Timer(f"{MARKET} fresh option margin")
    opt_margin_time.start()

    opt_cos = [(c, o) for c, o in zip(fresh_contracts, fresh_orders)]

    with IB().connect(HOST, PORT, CID) as ib:
        df_opt_margins = ib.run(
            executeAsync(
                ib=ib,
                algo=margin,
                cts=opt_cos,
                CONCURRENT=50 * 4,
                TIMEOUT=5,
                post_process=save_df,
                DATAPATH=DATAPATH,
                OP_FILENAME="",
                **{'FILL_DELAY': 5},
            )
        )

    opt_margin_time.stop()

    # * GET ROM AND SET EXPECTED OPTION PRICE

    # . integrate price, iv and margins
    df_fresh2 = (
        df_fresh1.set_index("conId")
        .join(df_opt_margins.set_index("conId")[["comm", "margin"]])
        .join(
            df_opt_prices.set_index("conId")[
                ["bid", "ask", "close", "last", "iv", "price"]
            ]
        )
        .reset_index()
    )

    # . update null iv with und_iv
    m_iv = df_fresh2.iv.isnull()
    df_fresh2.loc[m_iv, "iv"] = df_fresh2[m_iv].und_iv

    # . update calculated sd mult for strike
    df_fresh2.insert(19, "sdMult", calcsdmult_df(df_fresh2.strike, df_fresh2))

    # . put the order quantity
    df_fresh2["qty"] = 1 if MARKET == 'SNP' else df_fresh2.lot

    # . fill empty commissions
    if MARKET == 'NSE':
        commission = 20.0
    else:
        commission = 2.0

    df_fresh2['comm'].fillna(value=commission, inplace=True)

    # ... add intrinsic and time values
    df_fresh2 = df_fresh2.assign(intrinsic=np.where(df_fresh2.right == 'C',
                                                    (df_fresh2.undPrice -
                                                     df_fresh2.strike).clip(0, None),
                                                    (df_fresh2.strike - df_fresh2.undPrice).clip(0, None)))
    df_fresh2 = df_fresh2.assign(
        timevalue=df_fresh2.price - df_fresh2.intrinsic)

    # . compute rom based on timevalue, remove zero rom and down-sort on it
    df_fresh2["rom"] = (
        (df_fresh2.timevalue * df_fresh2.lot - df_fresh2.comm).clip(0)
        / df_fresh2.margin
        * 365
        / df_fresh2.dte
    )

    df_fresh2 = df_fresh2[df_fresh2.rom > 0]\
        .sort_values("rom", ascending=False)\
        .reset_index(drop=True)

    # .establish expRom
    #    ... for those whose RoM is < MINROM, make it equal to MINROM
    df_fresh2["expRom"] = np.maximum(ibp.MINEXPROM, df_fresh2.rom)

    # . set expPrice to be based on expRom
    df_fresh2["expPrice"] = (
        df_fresh2.expRom
        * np.maximum(ibp.MINOPTSELLPRICE, df_fresh2.price)
        / df_fresh2.rom
    ).apply(lambda x: get_prec(x, ibp.PREC))

    # . remove NaN from expPrice
    df_fresh2 = df_fresh2.dropna(subset=['expPrice']).reset_index(drop=True)

    # * PICKLE AND SAVE TO EXCEL
    df_fresh2.to_pickle(DATAPATH.joinpath("df_fresh.pkl"))

    df_calls = df_fresh2[df_fresh2.right == "C"].reset_index(drop=True)
    df_puts = df_fresh2[df_fresh2.right == "P"].reset_index(drop=True)

    # ... initiate Excel writer object
    writer = pd.ExcelWriter(DATAPATH.joinpath(
        "df_fresh.xlsx"), engine="xlsxwriter")

    df_fresh2.to_excel(
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
        print(f"\nError {e}: df_fresh.xlsx is open or has some issues!!!\n")

    fresh_time.stop()

    return df_fresh2


if __name__ == "__main__":

    # * USER INTERFACE

    # . first... get first user input for market
    mkt_dict = {1: "NSE", 2: "SNP"}
    mkt_ask_range = [i + 1 for i in list(range(len(mkt_dict)))]
    mkt_ask = "Create fresh naked options for:\n"
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

    get_fresh(MARKET=MARKET, RECALC_UNDS=True)
