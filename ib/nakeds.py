# Generate fresh nakeds

# ** SETUP
# .Imports
import os
import pathlib

import numpy as np
import pandas as pd
from ib_insync import IB, MarketOrder, util

from dfrq import get_dfrq
from engine import executeAsync, get_unds, margin, post_df, price
from support import (Timer, Vars, calcsdmult_df, fallrise, get_col_widths,
                     get_dte, get_market, get_prec, get_prob, yes_or_no)

# Set pandas display format
pd.options.display.float_format = "{:,.2f}".format


def get_nakeds(
    MARKET: str,
    RUN_ON_PAPER: bool = False,
    SYMBOL: str = "",  # Do nakeds for ALL symbols if SYMBOL is null
    EARLIEST: bool = False,  # Filter options only for the earliest DTE
    RECALC_UNDS: bool = True,
    SAVE: bool = True,  # If SYMBOL | EARLIEST, save will be turned OFF
    NKD_FILENAME: str = "",
) -> pd.DataFrame:

    # . start the time
    nakeds_time = Timer("nakeds")
    nakeds_time.start()

    ibp = Vars(MARKET.upper())  # IB Parameters from var.yml
    HOST, CID = ibp.HOST, ibp.CID
    if RUN_ON_PAPER:
        PORT = ibp.PAPER
    else:
        PORT = ibp.PORT

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

    LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")
    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    # * SETUP LOGS AND CLEAR THEM
    LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_nakeds.log")
    util.logToFile(path=LOGFILE, level=30)
    with open(LOGFILE, "w"):
        pass

    # * LOAD FILES
    df_symlots = pd.read_pickle(DATAPATH.joinpath("df_symlots.pkl"))
    df_chains = pd.read_pickle(DATAPATH.joinpath("df_chains.pkl"))
    df_ohlcs = pd.read_pickle(DATAPATH.joinpath("df_ohlcs.pkl"))
    df_opts = pd.read_pickle(DATAPATH.joinpath("df_opts.pkl"))

    # * DETERMINE TO SAVE OR NOT

    if (
        (SYMBOL == "") & (EARLIEST == False) & (SAVE == True)
    ):  # default for save all nakeds
        NKD_FILENAME = "df_nakeds.pkl"

    elif (SAVE == True) & (SYMBOL != "") & (NKD_FILENAME == ""):
        NKD_FILENAME = "df_nkd_" + SYMBOL + ".pkl"

    else:
        NKD_FILENAME = "df_nkd_temp.pkl"

    # * GET THE NAKED SYMBOLS

    # . run dfrq
    dfrq = get_dfrq(MARKET, RUN_ON_PAPER=RUN_ON_PAPER)

    # . collect the naked symbols
    if SYMBOL:
        nakeds = set([SYMBOL])
    else:
        nakeds = set(dfrq[dfrq.status == "naked"].symbol)

    # . remove BLACKLISTS
    nakeds = nakeds - set(ibp.BLACKLIST)

    # * CLIP THE DATAFRAMES TO GET ONLLY NAKEDS

    df_symlots = df_symlots[df_symlots.symbol.isin(nakeds)]
    df_ohlcs = df_ohlcs[df_ohlcs.symbol.isin(nakeds)]
    df_chains = df_chains[df_chains.symbol.isin(nakeds)]
    df_opts = df_opts[df_opts.symbol.isin(nakeds)]

    # . remove opt columns
    cols1 = [
        "conId",
        "symbol",
        "strike",
        "expiry",
        "secType",
        "right",
        "contract",
        "lot",
    ]
    df_opts = df_opts[cols1]

    # . compute dte
    df_opts = df_opts.assign(dte=df_opts.expiry.apply(get_dte))

    # * ABORT NAKEDS IF EMPTY

    if df_symlots.empty:
        print(
            f"\nSymbol: {SYMBOL} not in df_symlots.pkl.\n"
            + f"...if needed put the {SYMBOL} in var.yml -> SPECIALS\n"
        )
        return None  # !!! ABORT NAKEDS DUE TO MISSING SYMBOL

    # * HANDLE EARLIEST

    # . filter on dte
    if EARLIEST:  # keep only the earliest expiring dte for each symbol
        df_opts = df_opts[df_opts.dte == df_opts.groupby("symbol").dte.transform(min)]

    else:  # remove dtes between MINDTE and MAXDTE
        df_opts = df_opts[df_opts.dte.between(ibp.MINDTE, ibp.MAXDTE, inclusive=True)]

    # * INTEGRATE UNDS

    # . build relevant underying contract
    und_cts = df_symlots.contract.unique()

    # . check for redoing unds
    if RECALC_UNDS:
        df_unds = get_unds(MARKET, und_cts, RUN_ON_PAPER=RUN_ON_PAPER, SAVE=False)
    else:
        df_unds = pd.read_pickle(DATAPATH.joinpath("df_unds.pkl"))

    # . integrate undPrice and und_iv
    df_opts = (
        df_opts.set_index("symbol")
        .join(df_unds.set_index("symbol")[["iv", "undPrice"]])
        .reset_index()
    )

    # * DETERMINE STANDARD DEVIATIONS

    # . recalculate sdMult from iv gleaned from und
    df_opts = df_opts.assign(und_sd=calcsdmult_df(df_opts.strike, df_opts))

    # * INTEGRATE FALLRISE

    sym_dtes = df_opts.groupby("symbol").dte.unique().to_dict()

    fr = []
    for k, v in sym_dtes.items():
        for d in v:
            try:
                fr.append(fallrise(df_hist=df_ohlcs[df_ohlcs.symbol == k], dte=d))
            except IndexError:
                pass

    df_fr = pd.DataFrame(fr)

    """ # not using named tuple!
    fr = [
        fallrise(df_hist=df_ohlcs[df_ohlcs.symbol == k], dte=d)
        for k, v in sym_dtes.items()
        for d in v
    ]
    df_fr = pd.DataFrame(fr).rename(
        columns={0: "symbol", 1: "dte", 2: "rise", 3: "fall"}
    ) """

    # . integrate fallrise df with undPrice
    df_fr1 = (
        df_fr.set_index("symbol")
        .join(df_unds[["symbol", "undPrice"]].set_index("symbol"))
        .reset_index()
    )

    # . make the rights
    df_fr1 = pd.concat(
        [df_fr1.assign(right="C"), df_fr1.assign(right="P")], ignore_index=True
    )

    # . make the gross fallrise w.r.t undPrice
    df_fr1 = df_fr1.assign(
        fallrise=np.where(
            df_fr1.right == "C",
            df_fr1.undPrice + df_fr1.rise,
            df_fr1.undPrice - df_fr1.fall,
        )
    )

    # . integrate fallrise with nakeds
    frcols = ["symbol", "dte", "right"]
    df_opts = (
        df_opts.set_index(frcols)
        .join(df_fr1[frcols + ["fallrise"]].set_index(frcols))
        .reset_index()
    )

    # . get the fallrise standard deviation multiple
    df_opts = df_opts.assign(fr_sd=calcsdmult_df(df_opts.fallrise, df_opts))

    # Rename iv to und_iv
    df_opts = df_opts.rename(columns={"iv": "und_iv"})

    # * CLIP TO GET TARGET NAKEDS

    # . remove calls and puts above sdMult and against direction
    call_mask = (
        (df_opts.right == "C")
        & (df_opts.und_sd > ibp.CALLSTDMULT)
        & (df_opts.strike > df_opts.undPrice)
    )
    put_mask = (
        (df_opts.right == "P")
        & (df_opts.und_sd > ibp.PUTSTDMULT)
        & (df_opts.strike < df_opts.undPrice)
    )
    df_opts = df_opts[call_mask | put_mask].reset_index(drop=True)

    # . remove opts above MAXOPTQTY_SYM

    df_opts = df_opts.assign(
        value=np.where(df_opts.right == "C", -1 * df_opts.strike, df_opts.strike)
    ).sort_values(["symbol", "dte", "value"])

    df_nakeds = (
        df_opts[
            df_opts.groupby(["symbol", "right", "dte"]).cumcount() < ibp.MAXOPTQTY_SYM
        ]
        .drop("value", 1)
        .reset_index(drop=True)
    )

    # * GET NAKED PRICE, IV AND MARGIN

    # . make naked contract and orders
    nakeds_contracts = df_nakeds.contract.to_list()

    nakeds_orders = [
        MarketOrder("SELL", lot / lot)
        if MARKET.upper() == "SNP"
        else MarketOrder("SELL", lot)
        for lot in df_nakeds.lot
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
                post_process=post_df,
                DATAPATH=DATAPATH,
                **{"FILL_DELAY": 5.5},
                OP_FILENAME="",
                SHOW_TQDM=True,
            )
        )

    opt_cos = [(c, o) for c, o in zip(nakeds_contracts, nakeds_orders)]

    with IB().connect(HOST, PORT, CID) as ib:
        df_opt_margins = ib.run(
            executeAsync(
                ib=ib,
                algo=margin,
                cts=opt_cos,
                CONCURRENT=200,
                TIMEOUT=6.5,
                post_process=post_df,
                DATAPATH=DATAPATH,
                OP_FILENAME="",
                SHOW_TQDM=True,
                **{"FILL_DELAY": 6.48},
            )
        )

    # . integrate price, iv and margins
    df_nakeds = (
        df_nakeds.set_index("conId")
        .join(df_opt_margins.set_index("conId")[["comm", "margin"]])
        .join(
            df_opt_prices.set_index("conId")[
                ["bid", "ask", "close", "last", "iv", "price"]
            ]
        )
        .reset_index()
    )

    # . update null iv with und_iv
    m_iv = df_nakeds.iv.isnull()
    df_nakeds.loc[m_iv, "iv"] = df_nakeds[m_iv].und_iv

    # * GET EXPECTED PRICE AND ROM

    # . compute sdMult
    df_nakeds = df_nakeds.assign(sdMult=calcsdmult_df(df_nakeds.strike, df_nakeds))

    # . compute prop
    df_nakeds = df_nakeds.assign(prop=df_nakeds.sdMult.apply(get_prob))

    # . compute intrinsic values
    df_nakeds = df_nakeds.assign(
        intrinsic=np.where(
            df_nakeds.right == "C",
            (df_nakeds.undPrice - df_nakeds.strike).clip(0, None),
            (df_nakeds.strike - df_nakeds.undPrice).clip(0, None),
        )
    )

    # . compute time values
    df_nakeds = df_nakeds.assign(timevalue=df_nakeds.price - df_nakeds.intrinsic)

    # . compute rom based on timevalue, remove zero rom and down-sort on it
    df_nakeds["rom"] = (
        (df_nakeds.timevalue * df_nakeds.lot - df_nakeds.comm).clip(0)
        / df_nakeds.margin
        * 365
        / df_nakeds.dte
    )

    df_nakeds = (
        df_nakeds[df_nakeds.rom > 0]
        .sort_values("rom", ascending=False)
        .reset_index(drop=True)
    )

    # . establish expRom
    #    ... for those whose RoM is < MINEXPROM, make it equal to MINEXPROM
    df_nakeds["expRom"] = np.maximum(ibp.MINEXPROM, df_nakeds.rom)

    # . set expPrice to be based on expRom
    df_nakeds["expPrice"] = (
        df_nakeds.expRom
        * np.maximum(ibp.MINOPTSELLPRICE, df_nakeds.price)
        / df_nakeds.rom
    ).apply(lambda x: get_prec(x, ibp.PREC))

    # . get the remaining quantities
    remq = dfrq.set_index("symbol").remq.to_dict()
    df_nakeds = df_nakeds.assign(remq=df_nakeds.symbol.map(remq))

    # . put the order quantity
    df_nakeds["qty"] = 1 if MARKET == "SNP" else df_nakeds.lot

    cols2 = [
        "conId",
        "symbol",
        "dte",
        "right",
        "strike",
        "expiry",
        "secType",
        "contract",
        "lot",
        "und_iv",
        "undPrice",
        "und_sd",
        "fallrise",
        "fr_sd",
        "comm",
        "margin",
        "bid",
        "ask",
        "close",
        "last",
        "iv",
        "intrinsic",
        "timevalue",
        "price",
        "expPrice",
        "rom",
        "expRom",
        "sdMult",
        "prop",
        "remq",
        "qty",
    ]

    df_nakeds = df_nakeds[cols2]

    # * SAVE AND PICKLE

    if SAVE:

        if NKD_FILENAME[-4:].upper() != ".PKL":
            NKD_FILENAME = NKD_FILENAME + ".pkl"

        XL_FILE = NKD_FILENAME[:-4] + ".xlsx"

        df_nakeds.to_pickle(DATAPATH.joinpath(NKD_FILENAME))

        df_calls = df_nakeds[df_nakeds.right == "C"].reset_index(drop=True)
        df_puts = df_nakeds[df_nakeds.right == "P"].reset_index(drop=True)

        # ... initiate Excel writer object
        writer = pd.ExcelWriter(DATAPATH.joinpath(XL_FILE), engine="xlsxwriter")

        df_nakeds.to_excel(
            writer,
            sheet_name="All",
            float_format="%.2f",
            index=False,
            freeze_panes=(1, 1),
        )

        df_calls.to_excel(
            writer,
            sheet_name="Calls",
            float_format="%.2f",
            index=False,
            freeze_panes=(1, 1),
        )

        df_puts.to_excel(
            writer,
            sheet_name="Puts",
            float_format="%.2f",
            index=False,
            freeze_panes=(1, 1),
        )

        all_sheet = writer.sheets["All"]
        puts_sheet = writer.sheets["Calls"]
        calls_sheet = writer.sheets["Puts"]
        sheets = [all_sheet, puts_sheet, calls_sheet]

        col_width = get_col_widths(df_calls)  # set the column width

        for sht in sheets:
            # Hide all rows without data

            sht.set_default_row(hide_unused_rows=True)

            for i, width in enumerate(col_width):
                sht.set_column(i, i, width)

            sht.set_column("A:A", None, None, {"hidden": True})  # Hide conId
            sht.set_column("G:G", None, None, {"hidden": True})  # Hide secType
            sht.set_column("H:H", None, None, {"hidden": True})  # Hide contract

        try:
            writer.save()
        except Exception as e:
            print(f"\nError {e}: {XL_FILE} is open or has some issues!!!\n")

    nakeds_time.stop()

    return df_nakeds


if __name__ == "__main__":

    # Initialize
    NKD_FILENAME = ""

    MARKET = get_market("Create nakeds options for:")
    RUN_ON_PAPER = yes_or_no("Run on PAPER? ")
    ONE_SYMBOL = yes_or_no("For ONE symbol? ")
    if ONE_SYMBOL:
        SYMBOL = input(f"\nGive the name of symbol: ").upper()
    else:
        SYMBOL = ""

    EARLIEST = yes_or_no("Only EARLIEST DTE? ")

    RECALC_UNDS = yes_or_no("Want to recaculate underlyings? ")

    SAVE = yes_or_no("Do you want to output result to file? ")

    if SAVE:
        NKD_FILENAME = input(
            "Give Filename to save with `.pkl` extension (e.g. df_nakeds.pkl): "
        )
        if NKD_FILENAME == "":
            NKD_FILENAME = "df_nakeds.pkl"

    y = get_nakeds(
        MARKET=MARKET,
        RUN_ON_PAPER=RUN_ON_PAPER,
        RECALC_UNDS=RECALC_UNDS,
        SYMBOL=SYMBOL,
        EARLIEST=EARLIEST,
        SAVE=SAVE,
        NKD_FILENAME=NKD_FILENAME,
    )

    print(
        y.drop(
            [
                "conId",
                "contract",
                "secType",
                "comm",
                "lot",
                "close",
                "last",
                "intrinsic",
            ],
            1,
        )
    )
