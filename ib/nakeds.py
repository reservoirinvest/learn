# generates nakeds from df_opts
import pathlib

import numpy as np
import pandas as pd
from ib_insync import IB, MarketOrder, util

from dfrq import get_dfrq
from engine import Timer, Vars, get_unds, qpAsync
from support import (calcsdmult_df, get_col_widths, get_market, get_prec,
                     get_prob, yes_or_no, get_col_widths)
from openpyxl.utils import get_column_letter


def get_nakeds(MARKET: str, 
            SYMBOL: str='', 
            RUN_ON_PAPER: bool=False, 
            EARLIEST: bool=False, 
            RECALC_UNDS: bool=True,
            SAVE: bool=True) -> pd.DataFrame:

    # start the timer
    naked_time = Timer("Nakeds")
    naked_time.start()

    ibp = Vars(MARKET.upper())  # IB Parameters from var.yml

    # set and empty log file
    logf = pathlib.Path.cwd().joinpath('data', 'log', 'nakeds.log')
    util.logToFile(path=logf, level=30)
    with open(logf, "w"):
        pass

    datapath = pathlib.Path.cwd().joinpath('data', MARKET.lower())

    df_opts = pd.read_pickle(datapath.joinpath("df_opts.pkl"))
    df_unds = pd.read_pickle(datapath.joinpath("df_unds.pkl"))

    ## ** PREPARE RAW DATA

    # get dfrq
    dfrq = get_dfrq(MARKET, RUN_ON_PAPER=False)

    # collect the symbols without blacklist

    if SYMBOL: # If a symbol is given it becomes a deep-dive
        NAKEDS = set([SYMBOL])
        EARLIEST = True
        DEEPDIVE = True
    else:
        NAKEDS = set(dfrq[dfrq.status == "naked"].symbol) - set(ibp.BLACKLIST)
        DEEPDIVE = False

    df_raw = df_opts[df_opts.symbol.isin(NAKEDS)]

    # handle EARLIEST

    # . filter on dte
    if EARLIEST:  # keep only the earliest expiring dte for each symbol
        df_raw = df_raw[df_raw.dte == df_raw.groupby("symbol").dte.transform(min)]

    else:  # remove dtes between MINDTE and MAXDTE
        df_raw = df_raw[df_raw.dte.between(ibp.MINDTE, ibp.MAXDTE, inclusive=True)]

    # handle RECALC_UNDS
    und_cts = df_unds[df_unds.symbol.isin(df_raw.symbol.unique())].contract

    if RECALC_UNDS:
        df_unds = get_unds(MARKET, und_cts, RUN_ON_PAPER=RUN_ON_PAPER, SAVE=False)

    # update und_iv and undPrice from df_unds
    df_raw.set_index('symbol', inplace=True)
    df_raw.update(df_unds[['symbol', 'iv', 'undPrice']].rename(columns={'iv': 'und_iv'}).set_index('symbol'))
    df_raw.reset_index(inplace=True)

    # determine standard deviations
    df_raw = df_raw.assign(und_sd=calcsdmult_df(df_raw.strike, df_raw.rename(columns={'und_iv':'iv'})))

    # ** CLIP FOR TARGETS

    remq = dfrq.set_index('symbol').remq.to_dict()

    if not DEEPDIVE:
        
        # . remove calls and puts above sdMult and against direction
        call_mask = (
            (df_raw.right == "C")
            & (df_raw.und_sd > ibp.CALLSTDMULT)
            & (df_raw.strike > df_raw.undPrice)
        )
        put_mask = (
            (df_raw.right == "P")
            & (df_raw.und_sd > ibp.PUTSTDMULT)
            & (df_raw.strike < df_raw.undPrice)
        )
        
        df_raw = df_raw[call_mask | put_mask].reset_index(drop=True) 

        # integrate with remq
        df_raw = df_raw.set_index('symbol').join(dfrq.set_index('symbol').remq).reset_index()

        df_raw.loc[df_raw.right=='P','strike']*= -1

        s=(df_raw.sort_values('strike').groupby(['symbol','right'])
            .cumcount()
            .reindex(df_raw.index)
        )

        df_nakeds = df_raw[s<df_raw['symbol'].map(remq)].sort_values(['symbol','right']).reset_index(drop=True)
        df_nakeds.loc[df_nakeds.right=='P', 'strike'] *= -1

    else:
        df_nakeds = df_raw.assign(remq = df_raw.symbol.map(remq))


    # ** GET PRICE, IV AND MARGIN

    # price and iv
    with IB().connect(ibp.HOST, ibp.PORT, ibp.CID) as ib:
        df_pr = ib.run(qpAsync(ib, df_nakeds.contract, **{'FILL_DELAY': 5.5}))
        ib.disconnect()

    # margins
    orders = [MarketOrder("SELL", lot / lot)
                if MARKET.upper() == "SNP"
                    else MarketOrder("SELL", lot)
                for lot in df_nakeds.lot]

    opt_cos = [(c, o) for c, o in zip(df_nakeds.contract, orders)]

    from engine import executeAsync, margin, post_df

    with IB().connect(ibp.HOST, ibp.PORT, ibp.CID) as ib:

        ib.client.setConnectOptions('PACEAPI')

        df_mgn = ib.run(
            executeAsync(
                ib=ib,
                algo=margin,
                cts=opt_cos,
                CONCURRENT=200,
                TIMEOUT=6.5,
                post_process=post_df,
                DATAPATH=datapath,
                OP_FILENAME="",
                SHOW_TQDM=True,
                **{"FILL_DELAY": 6.48},
            )
        )

    # integrate price, iv and margins
    df_nakeds = df_nakeds.set_index("conId")\
                    .join(df_mgn.set_index("conId")[["comm", "margin"]])\
                        .join(df_pr.set_index("conId")\
                            [["bid", "ask", "close", "last", "iv", "price"]])\
                                .reset_index()

    # replace nan commissions with default values
    df_nakeds.comm.fillna(20 if MARKET == 'NSE' else 0.65, inplace=True)

    # update null iv with und_iv
    m_iv = df_nakeds.iv.isnull()
    df_nakeds.loc[m_iv, "iv"] = df_nakeds[m_iv].und_iv

    ## ** GET EXPECTED PRICE AND ROM

    # compute sdMult
    df_nakeds = df_nakeds.assign(sdMult=calcsdmult_df(df_nakeds.strike, df_nakeds))

    # compute prop
    df_nakeds = df_nakeds.assign(prop=df_nakeds.sdMult.apply(get_prob))

    # compute intrinsic values
    df_nakeds = df_nakeds.assign(
        intrinsic=np.where(
            df_nakeds.right == "C",
            (df_nakeds.undPrice - df_nakeds.strike).clip(0, None),
            (df_nakeds.strike - df_nakeds.undPrice).clip(0, None),
        )
    )

    # compute time values
    df_nakeds = df_nakeds.assign(timevalue=df_nakeds.price - df_nakeds.intrinsic)

    # compute rom based on timevalue, remove zero rom and down-sort on it
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

    # establish expRom
    #    ... for those whose RoM is < MINEXPROM, make it equal to MINEXPROM
    df_nakeds["expRom"] = np.maximum(ibp.MINEXPROM, df_nakeds.rom)

    # set expPrice to be based on expRom
    df_nakeds["expPrice"] = (
        df_nakeds.expRom
        * np.maximum(ibp.MINOPTSELLPRICE, df_nakeds.price)
        / df_nakeds.rom
    ).apply(lambda x: get_prec(x, ibp.PREC))

    # . put the order quantity
    df_nakeds["qty"] = 1 if MARKET == "SNP" else df_nakeds.lot

    # . determine maxFall and maxRise for the dte
    df_nakeds = df_nakeds.assign(maxFall = (df_nakeds.undPrice-df_nakeds.fall).clip(0, None), 
                maxRise = (df_nakeds.undPrice+df_nakeds.rise))

    cols2 = ["conId", "symbol", "dte", "right", "strike", "expiry", "secType", "contract", 
            "lot", "und_iv", "undPrice", "und_sd", "comm", "margin", "maxFall", "maxRise", "rsi",
            "bid", "ask", "close", "last", "iv", "intrinsic", "timevalue", "sdMult", "price", 
            "expPrice", "rom", "expRom", "prop", "remq", "qty"]

    df_nakeds = df_nakeds[cols2]

    # * SAVE AND PICKLE

    if SAVE:
        NKD_FILENAME = 'df_nakeds.pkl'

        if NKD_FILENAME[-4:].upper() != ".PKL":
            NKD_FILENAME = NKD_FILENAME + ".pkl"

        XL_FILE = NKD_FILENAME[:-4] + ".xlsx"

        df_nakeds.to_pickle(datapath.joinpath(NKD_FILENAME))

        df_calls = df_nakeds[df_nakeds.right == "C"].reset_index(drop=True)
        df_puts = df_nakeds[df_nakeds.right == "P"].reset_index(drop=True)

        # ... initiate Excel writer object
        writer = pd.ExcelWriter(datapath.joinpath(XL_FILE), engine="openpyxl")

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

        col_width = get_col_widths(df_calls)  # get the column width to set

        for sht in sheets:
            for i, width in enumerate(col_width):
                sht.column_dimensions[get_column_letter(i+1)].width = width
        
            for col in ['A', 'G', 'H']:
                sht.column_dimensions[col].hidden= True        

        try:
            writer.save()
        except Exception as e:
            print(f"\nError {e}: {XL_FILE} is open or has some issues!!!\n")

    naked_time.stop()

    return df_nakeds

if __name__ == "__main__":

    MARKET = get_market("Create nakeds options for:")
    RUN_ON_PAPER = yes_or_no("Run on PAPER? ")
    ONE_SYMBOL = yes_or_no("For ONE symbol? ")
    if ONE_SYMBOL:
        SYMBOL = input(f"\nGive the name of symbol: ").upper()
    else:
        SYMBOL = ""

    EARLIEST = yes_or_no("Only EARLIEST DTE? ")

    RECALC_UNDS = yes_or_no("Want to recalculate underlyings? ")

    SAVE = yes_or_no("Do you want to output result to file? ")

    df_nakeds = get_nakeds(
        MARKET=MARKET,
        RUN_ON_PAPER=RUN_ON_PAPER,
        RECALC_UNDS=RECALC_UNDS,
        SYMBOL=SYMBOL,
        EARLIEST=EARLIEST,
        SAVE=SAVE)

    print(df_nakeds.drop(['conId', 'contract', 'secType', 
            'comm', 'lot', 'close', 'last', 'intrinsic'], 1))
